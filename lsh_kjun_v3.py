'''
Copyright (c) 2011, Yahoo! Inc.
All rights reserved.

Redistribution and use of this software in source and binary forms, 
with or without modification, are permitted provided that the following 
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of Yahoo! Inc. nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of Yahoo! Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Python implementation of Andoni's e2LSH.  This version is fast because it
# uses Python hashes to implement the buckets.  The numerics are handled 
# by the numpy routine so this should be close to optimal in speed (although
# there is no control of the hash tables layout in memory.)

# This file implements the following classes
#  lsh_quad - the basic projection algorithm (on k-dimensional hash)
#  index - a group of L lsh_quad hashes
#  TestDataClass - a generic class for handling the raw data
 
# To use
#  Call this routine with the -histogram flag to create some random 
#    test data and to compute the nearest-neighbor distances
#  Load the .distance file that is produced into Matlab and compute
#    the d_nn and d_any histograms from the first and second columns
#    of the .distance data.
#  Use these histograms (and their bin positions) as input to the 
#    Matlab ComputeMPLSHParameters() routine. 
#  This gives you the optimum LSH parameters.  You can use these
#    values directly as parameters to this code.
#  You can use the -ktest, -ltest and -wtest flags to test the 
#    parameters.

# Prerequisites: Python version 2.6 (2.5 might work) and NumPy

# By Malcolm Slaney, Yahoo! Research

import random, numpy, numpy as np, pickle, os, operator, traceback, sys, math, time, ipdb
import numpy.random as ra, ipdb, cPickle as pickle, time, scipy.io as sio
import itertools          # For Multiprobe
import copy
from datetime import datetime
from kjunutils import *

#######################################################################
# Note, the data is always a numpy array of size Dx1.
#######################################################################

# This class just implements k-projections into k integers
# (after quantization) and then reducing that integer vector
# into a T1 and T2 hash.  Data can either be entered into a
# table, or retrieved.

def from_v1_to_v3(index1):
  index3 = index_quad(index1.w,index1.k,index1.l,index1.bits);
  index3.myIDs = index1.myIDs;
  l = index1.l;
  k = index1.k;
  d = index1.projections[0].projections_a.shape[1];
  projections_all = np.zeros((l*k,d+d**2),np.float32);
  assert projections_all.flags['C_CONTIGUOUS'] == True;
  for (i,p) in enumerate(index1.projections): # of tables
    assert(p.projections_b.flags['C_CONTIGUOUS'] == True);
    projections_all[(k*i):(k*(i+1)),:] = np.hstack( [p.projections_a, 
           p.projections_b.reshape(k,d**2)] );
    index3.projections[i].projections = None;
    index3.projections[i].buckets = p.buckets;
    index3.projections[i].binsStat = p.binsStat;
    index3.projections[i].rawProjs = p.rawProjs;
  index3.projections_all = projections_all;
  return index3;

def to_serializable(index3):
#def from_nonnext_to_next(index3):
  myDict = copy.copy(index3.__dict__);
  myDict['projections_all'] = myDict['projections_all'].tolist();
  myDict['projections'] = []; #copy.deepcopy(myDict['projections']);
  for p in index3.projections:
    innerDict = copy.copy(p.__dict__);
    innerDict['rawProjs'] = None;
    myDict['projections'].append(innerDict);

  return myDict;

def from_serializable(serializedObj):
  # prepare a new object
  newObj = copy.copy(serializedObj);
  projections = serializedObj['projections'];
  newObj['projections'] = [];
  for innerDict in projections:
    newLsh = lsh_quad(0,0);
    newLsh.__dict__ = innerDict;
    newObj['projections'].append(newLsh);
  myIndex = index_quad(0,0,0);
  myIndex.__dict__ = newObj;
  myIndex.projections_all = np.array(myIndex.projections_all, dtype=np.float32);
  return myIndex;


# def from_next_to_nonnext(index3):
#   index3.projections_all = np.array(index3.projections_all, dtype=np.float32);
#   return index3;


# FIXME what is T1/T2 hashing?
class lsh_quad:
  '''This class implements one k-dimensional projection, the T1/T2 hashing
  and stores the results in a table for later retrieval.  Input parameters
  are the bin width (w, floating point, or float('inf') to get binary LSH), 
  and the number of projections to compute for one table entry (k, an integer).'''
  def __init__(self, w, k, bits=64):
    self.k = k  # Number of projections
    self.w = w  # Bin width
#     self.projections_a = None
#     self.projections_b = None
    self.projections = None;
    self.buckets = {}
    self.binsStat = [{} for i in range(self.k)];
    self.rawProjs = {}
    self.bits = bits;

  #- KJUNQUAD
  def KeepProjectionsAndRepopulate_k(self, k):
    assert False, "not implemented"
    assert(self.k >= k);
    self.k = k;
    self.buckets = {};
    self.binsStat = [{} for i in range(self.k)];
    self.projections_a = self.projections_a[:self.k,:];
    self.projections_b = self.projections_b[:self.k,:,:];
    self.bias = self.bias[:self.k,:];
    self.rawProjs = dict( (k,v[:self.k,:]) for (k,v) in self.rawProjs.iteritems() );
    for idnum in sorted(self.rawProjs.keys()):
      self.InsertIntoTableFromCache(idnum);
  
  def KeepProjectionsAndReset(self, w):
    self.w = w;
    self.buckets = {};
    self.binStat = [{} for i in range(self.k)];

  def KeepProjectionsAndRepopulate(self, w):
    self.w = w;
    self.buckets = {};
    self.binsStat = [{} for i in range(self.k)];
    for idnum in sorted(self.rawProjs.keys()):
      self.InsertIntoTableFromCache(idnum);

  # This only works for Python >= 2.6
  def sizeof(self):
    '''Return how much storage is needed for this object. In bytes
    '''
    return sys.getsizeof(self.buckets) + \
      sys.getsizeof(self.projections) + \
      sys.getsizeof(self)
    
  # Create the random constants needed for the projections.
  # Can't do this until we see some data, so we know the 
  # diementionality.
  def CreateProjections(self, dim):
    self.dim = dim
#     self.projections_a = np.random.randn(self.k, self.dim)     #- KJUNQUAD
#     self.projections_b = np.random.randn(self.k, self.dim, self.dim)  #- KJUNQUAD
    self.projections = np.random.randn(self.k, self.dim + self.dim**2);
    self.bias = np.random.rand(self.k, 1)   #- FIXME populate the bias term
    if (self.bits == 32):
#       self.projections_a = self.projections_a.astype(np.float32);
#       self.projections_b = self.projections_b.astype(np.float32);
      self.projections = self.projections.astype(np.float32);
      self.bias = self.bias.astype(np.float32);
    if 0:
      print "Dim is", self.dim
      print 'Projections:\n', self.projections
    if 0:
      # Write out the project data so we can check it's properties.
      # Should be Gaussian with mean of zero and variance of 1.
      fp = open('Projections.data', 'w')
      for i in xrange(0,self.projections.shape[0]):
        for j in xrange(0,self.projections.shape[1]):
          fp.write('%g ' % self.projections[i,j])
        fp.write('\n')
      
  # Compute the t1 and t2 hashes for some data.  Doing it this way 
  # instead of in a loop, as before, is 10x faster.  Thanks to Anirban
  # for pointing out the flaw.  Not sure if the T2 hash is needed since
  # our T1 hash is so strong.
  debugFP = None
  firstTimeCalculateHashes = False    # Change to false to turn this off
  infinity = float('inf')          # Easy way to access this flag

  # KJUNQUAD the following is being called in insertintotable function
  def CalculateHashes(self, data):
    '''Multiply the projection data (KxD) by some data (Dx1), 
    and quantize'''
#    if self.projections_a is None:            # KJUNQUAD 
    if self.projections is None:            # KJUNQUAD 
      assert(data.shape[0] > 1 or data.shape[1] > 2);
      self.CreateProjections(max(data.shape[0], data.shape[1]))
    rawProj = np.zeros((self.k,1), 'float');  # KJUNQUAD
    bins = np.zeros((self.k,1), 'int')
    if self.w == lsh_quad.infinity:
      # Binary LSH
      rawProj[:] = self.CalcRawProjection(data);
      bins[:] = (np.sign(rawProj)+1)/2.0;
    else:
      rawProj[:] = self.CalcRawProjection(data);
      bins[:] = np.floor(self.bias + rawProj/self.w);

    t1 = self.ListHash(bins)
    t2 = self.ListHash(bins[::-1])    # Reverse data for second hash # FIXME what is t2? is it helping the multiprobe somehow?
    return t1, t2, bins, rawProj
#- KJUNQUAD: KJUNQUAD-1: to see how the trick works.
#  In [63]: aa = array([1,2],ndmin=2).transpose();
#  In [64]: aa
#  Out[64]: 
#  array([[1],
#         [2]])
#
#           A = ra.randn(3,2,2);
#  In [66]: A[1,:,:] = matrix("[1 2; 3 4]");
#  In [67]: A
#  Out[67]: 
#  array([[[ 0.39570502,  1.50488776],
#          [ 1.29082363, -0.60788434]],
#  
#         [[ 1.        ,  2.        ],
#          [ 3.        ,  4.        ]],
#  
#         [[ 0.19120285,  0.52742147],
#          [-1.08724174,  0.80852552]]])
#  
#  In [68]: dot(aa.transpose(),dot(A, aa))[0,:,:]
#  Out[68]: 
#  array([[  3.55559046],
#         [ 27.        ],
#         [  2.30566438]])
# 

  # KJUNQUAD
  def CalculateHashesBulk(self, dataMat):
    '''Multiply the projection data (KxD) by some data (Dx1), 
    and quantize'''
#    if self.projections_a is None:            # KJUNQUAD 
    if self.projections is None:            # KJUNQUAD 
      self.CreateProjections(dataMat.shape[0]);
    rawProjMat = self.CalcRawProjectionBulk(dataMat);
    N = dataMat.shape[1];
    binsMat = np.zeros((self.k,N), 'int');
    if self.w == lsh_quad.infinity:
      binsMat[:,:] = (np.sign(rawProjMat) + 1)/2.0;
    else:
      binsMat[:,:] = np.floor(self.bias + rawProjMat/self.w);
    t = np.zeros((2,N), 'int');
    for j in range(rawProjMat.shape[1]):
      t[0,j],t[1,j] = self.ListHashBoth(binsMat[:,j:j+1]);

    return t, binsMat, rawProjMat
#     rawProj = np.zeros((self.k,1), 'float');  # KJUNQUAD
#     bins = np.zeros((self.k,1), 'int')
#     if self.w == lsh_quad.infinity:
#       # Binary LSH
#       rawProj[:] = self.CalcRawProjection(data);
#       bins[:] = (np.sign(rawProj)+1)/2.0;
#     else:
#       rawProj[:] = self.CalcRawProjection(data);
#       bins[:] = np.floor(self.bias + rawProj/self.w);
# 
#     t1 = self.ListHash(bins)
#     t2 = self.ListHash(bins[::-1])    # Reverse data for second hash # FIXME what is t2? is it helping the multiprobe somehow?
#     return t1, t2, bins, rawProj

  # KJUNQUAD 
  def CalcRawProjectionForQuery(self, query):    
    if (self.bits == 32 and query.dtype is not np.float32): query = query.astype(np.float32);
    return np.dot(self.projections, query);

  #  return[i] is i-th projection: proj_a[i,:]*x + x'*proj_b[i,:,:]*x 
  def CalcRawProjection(self, data):    
    assert False, "not implemented"
    if (self.bits == 32): data = data.astype(np.float32); # in case of 32bit 
    return (np.dot(self.projections_a, data) + \
                np.dot(data.transpose(), np.dot(self.projections_b, data))[0,:,:]) # KJUNQUAD: see below for how the trick works 

  def CalcRawProjectionBulk(self, dataMat):    
    if (self.bits == 32): dataMat = dataMat.astype(np.float32); # in case of 32bit 
    ret = np.array((self.k, dataMat.shape[1]), ndmin=2);
#     for i in range(self.k):
#       ret[i,:] = np.dot(self.projections_a[i,:], dataMat) + \
#           np.dot(dataMat,T, np.dot(self.projections_b[i,:,:]
    return np.dot(self.projections_a, dataMat) + \
        np.sum(dataMat * np.dot(self.projections_b, dataMat), 1);
    
  #- FIXME: this is hash for finding hash of the key.
  # Input: A Nx1 array (of integers)
  # Output: A 28 bit hash value.
  # From: http://stackoverflow.com/questions/2909106/
  #  python-whats-a-correct-and-good-way-to-implement-hash/2909572#2909572
  def ListHash(self, d):
    # return str(d).__hash__()    # Good for testing, but not efficient
    if d is None or len(d) == 0:
      return 0
    # d = d.reshape((d.shape[0]*d.shape[1]))
    value = d[0, 0] << 7
    for i in d[:,0]:
      value = (101*value + i)&0xfffffff
    return value  

  def ListHashBoth(self, d):
    # return str(d).__hash__()    # Good for testing, but not efficient
    #if d is None or len(d) == 0:
    #  return 0
    # d = d.reshape((d.shape[0]*d.shape[1]))
    t1 = d[0, 0] << 7
    for i in d[:,0]:
      t1 = (101*t1 + i)&0xfffffff
    t2 = d[-1, 0] << 7
    for i in d[::-1,0]:
      t2 = (101*t2 + i)&0xfffffff
    return t1, t2
  
  # Just a debug version that returns the bins too.
  def CalculateHashes2(self, data):
    print("not implemented"); sys.exit(-1);
    if self.projections is None:
      print "CalculateHashes2: data.shape=%s, len(data)=%d" % (str(data.shape), len(data))
      self.CreateProjections(len(data))
    bins = np.zeros((self.k,1), 'int')
    parray = np.dot(self.projections, data)
    bins[:] = np.floor(parray/self.w  + self.bias)
    t1 = self.ListHash(bins)
    t2 = self.ListHash(bins[::-1])    # Reverse data for second hash
    # print self.projections, data, parray, bins
    # sys.exit(1)
    return t1, t2, bins, parray
    
  # Return a bunch of hashes, depending on the level of multiprobe 
  # asked for.  Each list entry contains T1, T2. This is a Python
  # iterator... so call it in a for loop.  Each iteration returns
  # a bin ID (t1,t2)
  # [Need to store bins in integer array so we don't convert to         % FIXME what does this mean?
  # longs prematurely and get the wrong hash!]
  def CalculateHashIterator(self, query, multiprobeRadius=0, queryRawProjection=None, probeUpto=True):
    assert(self.projections is None);
#     if self.projections is None: # KJUNQUAD
#       self.CreateProjections(len(query))
    bins = np.zeros((self.k,1), 'int')
    directVector = np.zeros((self.k,1), 'int')
    newProbe = np.zeros((self.k,1), 'int')
    if self.w == lsh_quad.infinity: # binary hashing
      if (queryRawProjection is not None): #- use the pre-computed raw projection
        points = queryRawProjection;
      else:
        assert False, "not implemented"
        points = self.CalcRawProjectionForQuery(query);
      bins[:] = (np.sign(points)+1)/2.0
      directVector[:] = -np.sign(bins-0.5)
    else:                           # gaussian hashing
      if (queryRawProjection is not None): #- use the pre-computed raw projection
        rawProj = queryRawProjection;
      else:
        rawProj = self.CalcRawProjectionForQuery(query);
      points = self.bias + rawProj/self.w;
      bins[:] = np.floor(points);
      directVector[:] = np.sign(points-np.floor(points)-0.5)      # FIXME for multiprove
    if (probeUpto): # do multiprove from 0 to multiprobeRadius
#       t1 = self.ListHash(bins)
#       t2 = self.ListHash(bins[::-1])
      t1,t2 = self.ListHashBoth(bins)
      yield (t1,t2)
      if multiprobeRadius > 0:
        dimensions = range(self.k)
        deltaVector = np.zeros((self.k, 1), 'int')  # Preallocate
        for r in range(1, multiprobeRadius+1):                         #FIXME In the end, this returns nearby buckets
          # http://docs.python.org/library/itertools.html
          for candidates in itertools.combinations(dimensions, r):     #FIXME do ?itertools.combinations; it returns all possible subset of size r
            deltaVector *= 0            # Start Empty                  #FIXME a trick to be fast
            deltaVector[list(candidates), 0] = 1  # Set some bits
            newProbe[:] = bins + deltaVector*directVector  # New probe  #FIXME elementwise product.
#             t1 = self.ListHash(newProbe)
#             t2 = self.ListHash(newProbe[::-1])    # Reverse query for second hash
            t1,t2 = self.ListHashBoth(newProbe)
            # print "Multiprobe probe:",newProbe, t1, t2
            yield (t1,t2)
    else: # KJUNQUAD  do multiprove for exactly `multiprobeRadius`
      if multiprobeRadius == 0:
#         t1 = self.ListHash(bins)
#         t2 = self.ListHash(bins[::-1])
        t1,t2 = self.ListHashBoth(bins)
        yield (t1,t2)
      else:
        dimensions = range(self.k)
        deltaVector = np.zeros((self.k, 1), 'int')  # Preallocate
        r = multiprobeRadius;
        # http://docs.python.org/library/itertools.html
        for candidates in itertools.combinations(dimensions, r):     
          deltaVector *= 0            # Start Empty                  
          deltaVector[list(candidates), 0] = 1  # Set some bits
          newProbe[:] = bins + deltaVector*directVector  # New probe  
#           t1 = self.ListHash(newProbe)
#           t2 = self.ListHash(newProbe[::-1])    # Reverse query for second hash
#           t1,t2 = self.ListHashBoth(newProbe)
#           yield (t1,t2)
          yield self.ListHashBoth(newProbe)
  
  # Put some data into the hash bucket for this LSH projection
  def InsertIntoTable(self, idnum, data):
    assert(len(data.shape) ==2 and data.shape[1] == 1)
    (t1, t2, bins, rawProj) = self.CalculateHashes(data)
    if t1 not in self.buckets:
      self.buckets[t1] = {t2: [idnum]}
    else:
      if t2 not in self.buckets[t1]:
        self.buckets[t1][t2] = [idnum]                 #FIXME hmm..... the bucket has 2-d index
      else:
        self.buckets[t1][t2].append(idnum)
    #- KJUNQUAD update bins stat
    for i in range(len(bins)):
      v = bins[i,0];
      if not self.binsStat[i].has_key(v):
        self.binsStat[i][v] = 0;
      self.binsStat[i][v] += 1;
    #- KJUNQUAD raw projections
    self.rawProjs[idnum] = rawProj;

  def InsertIntoTableBulk(self, dataMat):
    (t, binsMat, rawProjMat) = self.CalculateHashesBulk(dataMat) # t is the hashed key
    for ii in range(t.shape[1]):
      t1, t2 = t[0,ii],t[1,ii]
      bins = binsMat[:,ii:ii+1];
      rawProj = rawProjMat[:,ii:ii+1];
      if t1 not in self.buckets:
        self.buckets[t1] = {t2: [ii]}
      else:
        if t2 not in self.buckets[t1]:
          self.buckets[t1][t2] = [ii]                 #FIXME hmm..... the bucket has 2-d index
        else:
          self.buckets[t1][t2].append(ii)
      #- KJUNQUAD update bins stat
      for i in range(len(bins)):
        v = bins[i,0];
        if not self.binsStat[i].has_key(v):
          self.binsStat[i][v] = 0;
        self.binsStat[i][v] += 1;
      #- KJUNQUAD raw projections
      self.rawProjs[ii] = rawProj;

  # Put some data into the hash bucket for this LSH projection
  def InsertIntoTableFromCache(self, idnum):
    bins = np.zeros((self.k,1), 'int')
    if self.w == np.inf:
      bins[:] = (np.sign(self.rawProjs[idnum])+1)/2.0;
    else:
      bins[:] = np.floor(self.bias + self.rawProjs[idnum]/self.w);
    t1 = self.ListHash(bins)      
    t2 = self.ListHash(bins[::-1])
    # below is the same
    if t1 not in self.buckets:
      self.buckets[t1] = {t2: [idnum]}
    else:
      if t2 not in self.buckets[t1]:
        self.buckets[t1][t2] = [idnum]                 #FIXME hmm..... the bucket has 2-d index
      else:
        self.buckets[t1][t2].append(idnum)
    #- KJUNQUAD update bins stat
    for i in range(len(bins)):
      v = bins[i,0];
      if not self.binsStat[i].has_key(v):
        self.binsStat[i][v] = 0;
      self.binsStat[i][v] += 1;
  
  # Find some data in the hash bucket.  Return all the ids
  # that we find for this T1-T2 pair.
  def FindXXObsolete(self, data):
    print("not implemented"); sys.exit(-1);
#     (t1, t2, bins) = self.CalculateHashes(data)
#     if t1 not in self.buckets:
#       return []
#     row = self.buckets[t1]
#     if t2 not in row:
#       return []
#     return row[t2]
    
  # 
  def Find(self, data, multiprobeRadius=0):
    '''Find the points that are close to the query data.  Use multiprobe
    to also look in nearby buckets.'''
    res = []
    for (t1,t2) in self.CalculateHashIterator(data, multiprobeRadius): #FIXME this becomes a loop only if multiprobeRadius != 0
      # print "Find t1:", t1
      if t1 not in self.buckets:
        continue
      row = self.buckets[t1]
      if t2 not in row:
        continue
      res += row[t2]
    return res

  #- KJUNQUAD: distFunc must be (query, idnum) form where query is d+d^2 by 1 array and idnum is the index of the datapoint.
  #- also returns `nDistComp` which will tell us if we have exhausted the loop or not.
  #- provide debugHistMinCache=[] if you want to retrieve the history of minimum distance.
  def FindWithDist(self, query, distFunc, maxDistComp, debugHistMinCache=None):
    '''Find the points that are close to the query data.  Use multiprobe
    to also look in nearby buckets.'''
    nDistComp = 0;
    minDist = float('inf');
    multiprobeRadius = self.k;
    minId = -1;
    for (t1,t2) in self.CalculateHashIterator(query, multiprobeRadius): 
      # print "Find t1:", t1
      if t1 not in self.buckets:
        continue
      row = self.buckets[t1]
      if t2 not in row:
        continue
      for idnum in row[t2]:
        d = distFunc(query, idnum)[0,0];
        if (d <= minDist):
          minDist = d;
          minId = idnum;
        if (debugHistMinCache is not None):   # save the history for debugging
          debugHistMinCache.append(minDist);
        nDistComp += 1;
        if (nDistComp >= maxDistComp):
          return idnum, nDistComp;
    return minId, nDistComp;

  #- KJUNQUAD: distFunc must be (query, idnum) form where query is d+d^2 by 1 array and idnum is the index of the datapoint.
  #- also returns `nDistComp` which will tell us if we have exhausted the loop or not.
  #- provide debugHistMinCache=[] if you want to retrieve the history of minimum distance.
  def FindUpto(self, query, maxDistComp, foundSet, foundList, multiprobeRadius=0, queryRawProjection=None, lshId=None, invalidSet=set()):
    '''Find the points that are close to the query data.  Use multiprobe
    to also look in nearby buckets.'''
    foundAll = False;
    for (t1,t2) in self.CalculateHashIterator(query, multiprobeRadius, queryRawProjection, probeUpto=False): 
      # print "Find t1:", t1
      if t1 not in self.buckets:
        continue
      row = self.buckets[t1]
      if t2 not in row:
        continue
      for idnum in row[t2]:
        if (idnum not in foundSet) and (idnum not in invalidSet):
          #- FOUND!
          foundSet.add(idnum);
          foundList.append((idnum,multiprobeRadius,lshId));
          if (len(foundSet) >= maxDistComp):
            foundAll = True;
            break;
      if (foundAll):
        break;
    return;

  # Create a dictionary showing all the buckets an ID appears in
  def CreateDictionary(self, theDictionary, prefix):
    for b in self.buckets:    # Over all buckets
      w = prefix + str(b)
      for c in self.buckets[b]:# Over all T2 hashes
        for i in self.buckets[b][c]:#Over ids
          if not i in theDictionary:
            theDictionary[i] = [w]
          else:
            theDictionary[i] += w
    return theDictionary


  # Print some stats for these lsh_quad buckets
  def StatsXXX(self):
    maxCount = 0; sumCount = 0; 
    numCount = 0; bucketLens = [];
    for b in self.buckets:
      for c in self.buckets[b]:
        l = len(self.buckets[b][c])
        if l > maxCount: 
          maxCount = l
          maxLoc = (b,c)
          # print b,c,self.buckets[b][c]
        sumCount += l
        numCount += 1
        bucketLens.append(l)
    theValues = sorted(bucketLens)
    med = theValues[(len(theValues)+1)/2-1]
    print "Bucket Counts:"
    print "\tTotal indexed points:", sumCount
    print "\tT1 Buckets filled: %d/%d" % (len(self.buckets), 0)
    print "\tT2 Buckets used: %d/%d" % (numCount, 0)
    print "\tMaximum T2 chain length:", maxCount, "at", maxLoc
    print "\tAverage T2 chain length:", float(sumCount)/numCount
    print "\tMedian T2 chain length:", med
  
  def HealthStats(self):
    '''Count the number of points in each bucket (which is currently
    a function of both T1 and T2)'''
    maxCount = 0; numCount = 0; totalIndexPoints = 0; 
    for b in self.buckets:
      for c in self.buckets[b]:
        l = len(self.buckets[b][c])
        if l > maxCount: 
          maxCount = l
          maxLoc = (b,c)
          # print b,c,self.buckets[b][c]
        totalIndexPoints += l
        numCount += 1
    T1Buckets = len(self.buckets)
    T2Buckets = numCount
    T1T2BucketAverage = totalIndexPoints/float(numCount)
    T1T2BucketMax = maxCount
    return (T1Buckets, T2Buckets, T1T2BucketAverage, T1T2BucketMax)
    
  # Get a list of all IDs that are contained in these hash buckets
  def GetAllIndices(self):
    theList = []
    for b in self.buckets:
      for c in self.buckets[b]:
        theList += self.buckets[b][c]
    return theList

  # Put some data into the hash table, see how many collisions we get.
  def Test(self, n):
    self.buckets = {}
    self.projections = None
    d = np.array([.2,.3])
    for i in range(0,n):
      self.InsertIntoTable(i, d+i)
    for i in range(0,n):
      r = self.Find(d+i)
      matches = sum(map(lambda x: x==i, r))
      if matches == 0:
        print "Couldn't find item", i
      elif matches == 1:
        pass
      if len(r) > 1: 
        print "Found big bin for", i,":", r
  

# Put together several LSH projections to form an index.  The only 
# new parameter is the number of groups of projections (one LSH class
# object per group.)
class index_quad:
  def __init__(self, w, k, l,bits=64):
    self.k = k; 
    self.l = l
    self.w = w
    self.projections = []
    self.projections_all = None;
    self.myIDs = []
    self.bits = bits;
    for i in range(0,l):  # Create all LSH buckets
      self.projections.append(lsh_quad(w, k,bits=bits))

#   def set_w(self, w):
#     self.w = w;
#     for i in range(0,l):
#       self.w = w;

  def KeepProjectionsAndRepopulate_k(self, k):
    assert(self.k >= k);
    self.k = k;
    for p in self.projections:
      p.KeepProjectionsAndRepopulate_k(k);

  #- KJUNQUAD
  def KeepProjectionsAndReset(self, w):
    self.w = w;
    for i in range(self.l):
      self.projections[i].KeepProjectionsAndReset(w);

  #- KJUNQUAD
  def KeepProjectionsAndRepopulate(self, w):
    self.w = w;
    for i in range(self.l):
      self.projections[i].KeepProjectionsAndRepopulate(w);

  # Only works for Python > 2.6
  def sizeof(self):
    '''Return the sizeof this index in bytes.
    '''
    return sum(p.sizeof() for p in self.projections) + \
      sys.getsizeof(self)

  # Replace idnum we are given with a numerical idnum.  Since we are going 
  # to use the ID in L tables, it is better to replace it here with
  # an integer.  We store the original ID in an array, and return it 
  # to the user when we do a find().
  def AddIDToIndex(self, idnum):
    print("not needed for us"); sys.exit(-1);
    if type(idnum) == int:
      return idnum        # Don't bother if already an int
    self.myIDs.append(idnum)
    return len(self.myIDs)-1
  
  def FindID(self, idnum):
    return idnum;
#     if type(idnum) != int or idnum < 0 or idnum >= len(self.myIDs):
#       return idnum
#     return self.myIDs[idnum]
  
  # Insert some data into all LSH buckets
  def InsertIntoTable(self, idnum, data):
    assert False, "not implemented";
#    intID = self.AddIDToIndex(idnum)   # KJUNQUAD this is not useful for our case.
    for p in self.projections:
      p.InsertIntoTable(idnum, data)

  def InsertIntoTableBulk(self, dataMat):
    assert False, "not implemented";
#    intID = self.AddIDToIndex(idnum)   # KJUNQUAD this is not useful for our case.
    for p in self.projections:
      p.InsertIntoTableBulk(dataMat)

  # KJUNQUAD
  def InsertIntoTableFromCache(self, idnum):
    assert False, "not implemented";
#    intID = self.AddIDToIndex(idnum)
#    intID = self.myIds....
    for p in self.projections:
      p.InsertIntoTableFromCache(idnum)

  def FindXXObsolete(self, data):
    '''Find some data in all the LSH buckets. Return a list of
    data's idnum and bucket counts'''
    items = [p.Find(data) for p in self.projections]
    results = {}
    for itemList in items: 
      for item in itemList:
        if item in results:      # Much faster without setdefault
          results[item] += 1
        else:
          results[item] = 1
    s = sorted(results.items(), key=operator.itemgetter(1), \
      reverse=True)
    return [(self.FindID(i),c) for (i,c) in s]
  
  def Find(self, queryData, multiprobeR=0):
    '''Find some data in all the LSH tables.  Use Multiprobe, with 
    the given radius, to search neighboring buckets.  Return a list of
    results.  Each result is a tuple consisting of the candidate ID
    and the number of times it was found in the index.'''
    results = {}
    for p in self.projections:
      ids = p.Find(queryData, multiprobeR)
      # print "Got back these IDs from p.Find:", ids
      for idnum in ids:
        if idnum in results:
          results[idnum] += 1
        else:
          results[idnum] = 1
    s = sorted(results.items(), key=operator.itemgetter(1), \
      reverse=True)
    return [(self.FindID(i),c) for (i,c) in s] # c is count; how many times it was retrieved.

  #- KJUNQUAD: distFunc must be (query, idnum) form where query is d+d^2 by 1 array and idnum is the index of the datapoint.
  #- also returns `nDistComp` which will tell us if we have exhausted the loop or not.
  #- provide debugHistMinCache=[] if you want to retrieve the history of minimum distance.
  def FindWithDist(self, queryData, distFunc, maxDistComp=100, debugHistMinCache=None):
    assert(len(self.projections) == 1) # I am skipping this general case for time management.
    if self.bits == 32: queryData = queryData.astype(np.float32);
    for p in self.projections:
      minId, nDistComp = p.FindWithDist(queryData, distFunc, maxDistComp, \
          debugHistMinCache=debugHistMinCache)
    return minId, nDistComp

  def FindUpto_old(self, query, maxDistComp, randomize=False, invalidSet=set()):
    if self.bits == 32: query = query.astype(np.float32);  # conver to 32 bit if necessary
    queryRawProjectionAry = [];
    for p in self.projections:
      queryRawProjectionAry.append( p.CalcRawProjectionForQuery(query) );
    foundSet = set();
    foundList = [];
    for multiprobeRadius in range(0,self.k+1):
      idxAry = range(len(self.projections));
      if (randomize):
        idxAry = ra.permutation(len(self.projections));
      for i in idxAry:
        p = self.projections[i];
        p.FindUpto(query, maxDistComp, foundSet, foundList, \
                   multiprobeRadius, queryRawProjectionAry[i], lshId=i, invalidSet=invalidSet)
        if (len(foundSet) >= maxDistComp):
          break;
      if (len(foundSet) >= maxDistComp):
        break;
    return foundSet, foundList

  #- in v3, we compute projections all at the same time.
  def FindUpto(self, query, maxDistComp, randomize=False, invalidSet=set()):
    if self.bits == 32: query = query.astype(np.float32);  # conver to 32 bit if necessary
#     queryRawProjectionAry = [];
#     for p in self.projections:
#       queryRawProjectionAry.append( p.CalcRawProjectionForQuery(query) );
    queryRawProjectionAry = np.dot(self.projections_all, query);
    foundSet = set();
    foundList = [];
    for multiprobeRadius in range(0,self.k+1):
      idxAry = range(self.l);
      if (randomize):
        idxAry = ra.permutation(self.l);
      for i in idxAry:
        p = self.projections[i];
        p.FindUpto(query, maxDistComp, foundSet, foundList,
            multiprobeRadius, queryRawProjectionAry[(self.k*i):(self.k*(i+1)),:],
            lshId=i, invalidSet=invalidSet)
        if (len(foundSet) >= maxDistComp):
          break;
      if (len(foundSet) >= maxDistComp):
        break;
    return foundSet, foundList

  def hello(self):
    print('hello');
    
  def FindExact(self, queryData, GetData, multiprobeR=0):
    '''Return a list of results sorted by their exact 
    distance from the query.  GetData is a function that
    returns the original data given its key.  This function returns
    a list of results, each result has the candidate ID and distance.'''
    s = self.Find(queryData, multiprobeR)
    # print "Intermediate results are:", s
    d = map(lambda (idnum,count): (idnum,((GetData(idnum)-queryData)**2).sum(), \
        count), s)
    s = sorted(d, key=operator.itemgetter(1))
    return [(self.FindID(i),d) for (i,d,c) in s]
  
  # Put some data into the hash tables.
  def Test(self, n):
    d = np.array([.2,.3])
    for i in range(0,n): 
      self.InsertIntoTable(i, d+i)
    for i in range(0,n):
      r = self.Find(d+i)
      print r
  
  # Print the statistics of each hash table.
  def Stats(self):
    for i in range(0, len(self.projections)):
      p = self.projections[i]
      print "Buckets", i, 
      p.Stats()

  # Get al the IDs that are part of this index.  Just check one hash
  def GetAllIndices(self):
    if self.projections and len(self.projections) > 0:
      p = self.projections[0]
      return p.GetAllIndices()
    return None
      
  # Return the buckets (t1 and t2 hashes) associated with a data point
  def GetBuckets(self, data):
    b = []
    for p in self.projections:
      ( t1, t2, bins, parray) = p.CalculateHashes2(data)
      print "Bucket:", t1, t2, bins, parray
      b += (t1, t2)
    return b
  
  # 
  def DictionaryPrefix(self, pc):
    prefix = 'W'
    prefixes = 'abcdefghijklmnopqrstuvwxyz'
    while pc > 0:  # Create unique ID for theis bucket
      prefix += prefixes[pc%len(prefixes)]
      pc /= len(prefixes)
    return prefix
    
  # Create a list ordered by ID listing which buckets are used for each ID
  def CreateDictionary(self):
    theDictionary = {}
    pi = 0
    for p in self.projections:
      prefix = self.DictionaryPrefix(pi)
      theDictionary = p.CreateDictionary(theDictionary,\
        prefix)
      pi += 1
    return theDictionary
  
  # Find the bucket ids that best correspond to this piece of data.
  def FindBuckets(self, data):
    theWords = []
    pi = 0
    for p in self.projections:
      prefix = self.DictionaryPrefix(pi)
      ( t1, t2, bins, parray) = p.CalculateHashes2(data)
      word = prefix + str(t1)
      theWords += [word]
      pi += 1
    return theWords
  
# Save an LSH index to a pickle file.
def SaveIndex(filename, ind):
  try:
    fp = open(filename, 'w')
    pickle.dump(ind, fp)
    fp.close()
    statinfo = os.stat(filename,)
    if statinfo:
      print "Wrote out", statinfo.st_size, "bytes to", \
        filename
  except:
    print "Couldn't pickle index to file", filename
    traceback.print_exc(file=sys.stderr)

# def SaveIndexGzip(filename, ind):
#   try:
#     fp = gzip.GzipFile(filename, 'w')
#     pickle.dump(ind, fp)
#     fp.close()
#     statinfo = os.stat(filename,)
#     if statinfo:
#       print "Wrote out", statinfo.st_size, "bytes to", \
#         filename
#   except:
#     print "Couldn't pickle index to file", filename
#     traceback.print_exc(file=sys.stderr)

# Read an LSH index from a pickle file.  
def LoadIndex(filename):
  if type(filename) == str:
    try:
      fp = open(filename, 'r')
    except:
      print "Couldn't open %s to read LSH Index" % (filename)
      return None
  else:
    fp = filename
  try:
    ind = pickle.load(fp)
    fp.close()
    return ind
  except:
    print "Couldn't read pickle file", filename
    traceback.print_exc(file=sys.stderr)

  


class TestDataClass:
  '''A bunch of routines used to generate data we can use to test
  this LSH implementation.'''
  def __init__(self):
    self.myData = None
    self.myIndex = None
    self.nearestNeighbors = {}    # A dictionary pointing to IDs
  
  def LoadData(self, filename):
    '''Load data from a flat file, one line per data point.'''
    lineCount = 0
    try: 
      fp = open(filename)
      if fp:
        for theLine in fp:      # Count lines in file
          if theLine == '':
            break
          lineCount += 1
        dim = len(theLine.split())  # Allocate the storage array
        self.myData = np.zeros((dim, lineCount))
        fp.seek(0,0)        # Go back to beginning of file
        lineCount = 0
        for theLine in fp:      # Now load the data
          data = [float(i) for i in theLine.split()]
          self.myData[:,lineCount] = data
          lineCount += 1
        fp.close()
      else:
        print "Can't open %s to LoadData()" % filename
    except:
      print "Error loading data from %s in TestDataClass.LoadData()" \
        % filename
      traceback.print_exc(file=sys.stderr)
    print "self.myData has %d lines and is:" % lineCount, self.myData
        
  def SaveData(self, filename):
    '''Save this data in a flat file.  One line per data point.'''
    numDims = self.NumDimensions()
    try:
      fp = open(filename, 'w')
      if fp:
        for i in xrange(0, self.NumPoints()):
          data = self.RetrieveData(i).reshape(numDims)
          fp.write(' '.join([str(d) for d in data]) + '\n')
        fp.close()
        return
    except:
      pass
    sys.stderr.write("Can't write test data to %s\n" % filename)
    
  def CreateIndex(self, w, k, l):
    '''Create an index for the data we have in our database.  Inputs are
    the LSH parameters: w, k and l.'''
    self.myIndex = index_quad(w, k, l)
    itemCount = 0
    tic = time.clock()
    for itemID in self.IterateKeys():
      features = self.RetrieveData(itemID)
      if features != None:
        self.myIndex.InsertIntoTable(itemID, features)
        itemCount += 1
    print "Finished indexing %d items in %g seconds." % \
      (itemCount, time.clock()-tic)
    sys.stdout.flush()
  
  def RetrieveData(self, idnum):
    '''Find a point in the array of data.'''
    idnum = int(idnum)            # Key in this base class is an int!
    if idnum < self.myData.shape[1]:
      return self.myData[:,idnum:idnum+1]
    return None
    
  def NumPoints(self):
    '''How many data point are in this database?'''
    return self.myData.shape[1]
    
  def NumDimensions(self):
    '''What is the dimensionality of the data?'''
    return self.myData.shape[0]
    
  def GetRandomQuery(self):
    '''Pick a random query from the dataset.  Return a key.'''
    return random.randrange(0,self.NumPoints())  # Pick random query
    
  def FindNearestNeighbors(self, count):
    '''Exhaustive search for count nearest-neighbor results.
    Save the results in a dictionary.'''
    numPoints = self.NumPoints()
    self.nearestNeighbors = {}
    for i in xrange(0,count):
      qid = self.GetRandomQuery()        # Pick random query
      qData = self.RetrieveData(qid)        # Find it's data
      nearestDistance2 = None
      nearestIndex = None
      for id2 in self.IterateKeys():
        if qid != id2:
          d2 = ((self.RetrieveData(id2)-qData)**2).sum()
          if idnum == -1:          # Debugging
            print qid, id2, qData, self.RetrieveData(id2), d2
          if nearestDistance2 is None or d2 < nearestDistance2:
            nearestDistance2 = d2
            nearestIndex = id2
      self.nearestNeighbors[qid] = \
        (nearestIndex, math.sqrt(nearestDistance2))
      if qid == -1:
        print qid, nearestIndex, math.sqrt(nearestDistance2)
        sys.stdout.flush()
  
  def SaveNearestNeighbors(self, filename):
    '''Save the nearest neighbor dictionary in a file.  Each line
    of the file contains the query key, the distance to the nearest
    neighbor, and the NN key.'''
    if filename.endswith('.gz'):
      import gzip
      fp = gzip.open(filename, 'w')
    else:
      fp = open(filename, 'w')
    if fp:
      for (query,(nn,dist)) in self.nearestNeighbors.items():
        fp.write('%s %g %s\n' % (str(query), dist, str(nn)))
      fp.close()
    else:
      print "Can't open %s to write nearest-neighbor data" % filename
  
  def LoadNearestNeighbors(self, filename):
    '''Load a file full of nearest neighbor data.'''
    self.nearestNeighbors = {}
    if filename.endswith('.gz'):
      import gzip
      fp = gzip.open(filename, 'r')
    else:
      fp = open(filename, 'r')
    if fp:
      print "Loading nearest-neighbor data from:", filename
      for theLine in fp:
        (k,d,nn) = theLine.split()
        if type(self.myData) == np.ndarray: # Check for array indices
          k = int(k)
          nn = int(nn)
          if k < self.NumPoints() and nn < self.NumPoints():
            self.nearestNeighbors[k] = (nn,float(d))
        elif k in self.myData and nn in self.myData:  # dictionary index
          self.nearestNeighbors[k] = (nn,float(d))
      fp.close()
      print " Loaded %d items into the nearest-neighbor dictionary." % len(self.nearestNeighbors)
    else:
      print "Can't open %s to read nearest neighbor data." % filename
          
  def IterateKeys(self):
    '''Iterate through all possible keys in the dataset.'''
    for i in range(self.NumPoints()):
      yield i
  
  def FindMedian(self):
    numDim = self.NumDimensions()
    numPoints = self.NumPoints()
    oneColumn = np.zeros((numPoints))
    medians = np.zeros((numDim))
    for d in xrange(numDim):
      rowNumber = 0
      for k in self.IterateKeys():
        oneData = self.RetrieveData(k)
        oneColumn[rowNumber] = oneData[d]
        rowNumber += 1
      m = np.median(oneColumn, overwrite_input=True)
      medians[d] = m
    return medians
    
  def ComputeDistanceHistogram(self, fp = sys.stdout):
    '''Calculate the nearest-neighbor and any-neighbor distance
    histograms needed for the LSH Parameter Optimization.  For
    a number of random query points, print the distance to the 
    nearest neighbor, and to any random neighbor.  This becomes 
    the input for the parameter optimization routine.  Enhanced
    to also print the NN binary projections.'''
    numPoints = self.NumPoints()
    # medians = self.FindMedian()    # Not used now, but useful for binary quantization
    print "Pulling %d items from the NearestNeighbors list for ComputeDistanceHistogram" % \
      len(self.nearestNeighbors.items())
    for (queryKey,(nnKey,nnDist)) in self.nearestNeighbors.items():
      randKey = self.GetRandomQuery()
      
      queryData = self.RetrieveData(queryKey)
      nnData = self.RetrieveData(nnKey)
      randData = self.RetrieveData(randKey)
      if len(queryData) == 0 or len(nnData) == 0:      # Missing, probably because of subsampling
        print "Skipping %s/%s because data is missing." % (queryKey, nnKey)
        continue
      anyD2 = ((randData-queryData)**2).sum()
      
      projection = np.random.randn(1, queryData.shape[0])
      # print "projection:", projection.shape
      # print "queryData:", queryData.shape
      # print "nnData:", nnData.shape
      # print "randData:", randData.shape
      queryProj = np.sign(np.dot(projection, queryData))
      nnProj = np.sign(np.dot(projection, nnData))
      randProj = np.sign(np.dot(projection, randData))
      
      # print 'CDH:', queryProj, nnProj, randProj
      fp.write('%g %g %d %d\n' % \
        (nnDist, math.sqrt(anyD2), \
         queryProj==nnProj, queryProj==randProj))
      fp.flush()      
    
  def ComputePnnPany(self, w, k, l, multiprobe=0):
    '''Compute the probability of Pnn and Pany for a given index size.
    Create the desired index, populate it with the data, and then measure
    the NN and ANY neighbor retrieval rates.
    Return 
      the pnn rate for one 1-dimensional index (l=1),
      the pnn rate for an l-dimensional index, 
      the pany rate for one 1-dimensional index (l=1), 
      and the pany rate for an l-dimensional index
      the CPU time per query (seconds)'''
    numPoints = self.NumPoints()
    numDims = self.NumDimensions()
    self.CreateIndex(w, k, l)      # Put data into new index
    cnn  = 0; cnnFull  = 0
    cany = 0; canyFull = 0
    queryCount = 0              # Probe the index
    totalQueryTime = 0
    startRecallTestTime = time.clock()
    # print "ComputePnnPany: Testing %d nearest neighbors." % len(self.nearestNeighbors.items())
    for (queryKey,(nnKey,dist)) in self.nearestNeighbors.items():
      queryData = self.RetrieveData(queryKey)
      if queryData is None or len(queryData) == 0:
        print "Can't find data for key %s" % str(queryKey)
        sys.stdout.flush()
        continue
      startQueryTime = time.clock()  # Measure CPU time
      matches = self.myIndex.Find(queryData, multiprobe)
      totalQueryTime += time.clock() - startQueryTime
      for (m,c) in matches:
        if nnKey == m:        # See if NN was found!!!
          cnn += c
          cnnFull += 1
        if m != queryKey:      # Don't count the query
          cany += c
      canyFull += len(matches)-1    # Total candidates minus 1 for query
      queryCount += 1
      # Some debugging for k curve.. print individual results
      # print "ComputePnnPany Debug:", w, k, l, len(matches), numPoints, cnn, cnnFull, cany, canyFull
    recallTestTime = time.clock() - startRecallTestTime
    print "Tested %d NN queries in %g seconds." % (queryCount, recallTestTime)
    sys.stdout.flush()
    if queryCount == 0:
      queryCount = 1          # To prevent divide by zero
    perQueryTime = totalQueryTime/queryCount
    print "CPP:", cnn, cnnFull, cany, canyFull
    print "CPP:",  cnn/float(queryCount*l), cnnFull/float(queryCount), \
      cany/float(queryCount*l*numPoints), canyFull/float(queryCount*numPoints), \
      perQueryTime, numDims
    return cnn/float(queryCount*l), cnnFull/float(queryCount), \
      cany/float(queryCount*l*numPoints), canyFull/float(queryCount*numPoints), \
      perQueryTime, numDims

  def ComputePnnPanyCurve(self, wList = .291032, multiprobe=0):
      if type(wList) == float or type(wList) == int:
        wList = [wList*10**((i-10)/10.0) for i in range(0,21)]
      for w in wList:
        (pnn, pnnFull, pany, panyFull, queryTime, numDims) = self.ComputePnnPany(w, 1, 10, multiprobe)
        if w == wList[0]:
          print "# w pnn pany queryTime"
        print "PnnPany:", w, multiprobe, pnn, pany, queryTime
        sys.stdout.flush()

  def ComputeKCurve(self, kList, w = .291032, r=0):
    '''Compute the number of ANY neighbors as a function of
    k.  Should go down exponentially.'''
    numPoints = self.NumPoints()
    l = 10
    for k in sorted(list(kList)):
      (pnn, pnnFull, pany, panyFull, queryTime, numDims) = self.ComputePnnPany(w, k, l, r)
      print w, k, l, r, pnn, pany, pany*numPoints, queryTime
      sys.stdout.flush()

  def ComputeLCurve(self, lList, w = 2.91032, k=10, r=0):
    '''Compute the probability of nearest neighbors as a function
    of l.'''
    numPoints = self.NumPoints()
    firstTime = True
    for l in sorted(list(lList)):
      (pnn, pnnFull, pany, panyFull, queryTime, numDims) = self.ComputePnnPany(w, k, l, r)
      if firstTime:
        print "# w k l r pnnFull, panyFull panyFull*N queryTime"
        firstTime = False
      print w, k, l, r, pnnFull, panyFull, panyFull*numPoints, queryTime
      sys.stdout.flush()

  
  
class RandomTestData(TestDataClass):
  '''Generate uniform random data points between -1 and 1.'''
  def CreateData(self, numPoints, dim):
    self.myData = (np.random.rand(dim, numPoints)-.5)*2.0

class HyperCubeTestData(TestDataClass):
  '''Create a hypercube of data.  All points are in the corners'''
  def CreateData(self, numDim, noise = None):
    numPoints = 2**numDim
    self.myData = np.zeros((numPoints, numDim))
    for i in range(0,numPoints):
      for b in range(0,numDim):
        if (2**b) & i:
          self.myData[b, i] = 1.0
    if noise != None:
      self.myData += (np.random.rand(numDim, numPoints)-.5)*noise

class RegularTestData(TestDataClass):
  '''Fill the 2-D test array with a regular grid of points between -1 and 1'''
  def CreateData(self, numDivs):
    self.myData = np.zeros(((2*numDivs+1)**2,2))
    i = 0
    for x in range(-numDivs, numDivs+1):
      for y in range(-numDivs, numDivs+1):
        self.myData[0, i] = x/float(divs)
        self.myData[1, i] = y/float(divs)
        i += 1
  
        

# Use Dimension Doubling to measure the dimensionality of a random
# set of data.  Generate some data (either random Gaussian or a grid)
# Then count the number of points that fall within the given radius of this 
# query.
def XXXTestDimensionality2():
  binWidth = .5
  if True:
    numPoints = 100000
    myTestData = TestDataClass(numPoints, 3)  
  else:
    myTestData = RegularTestData(100)
    numPoints = myTestData.NumPoints
  k = 4; l = 2; N = 1000
  myTestIndex = index_quad(binWidth, k, l, N)
  for i in range(0,numPoints):
    myTestIndex.InsertIntoTable(i, myTestData.RetrieveData(i))
  rBig = binWidth/8.0
  rSmall = rBig/2.0
  cBig = 0.0; cSmall = 0.0
  for idnum in random.sample(ind.GetAllIndices(), 2):
    qp = FindLSHTestData(idnum)
    cBig += myTestIndex.CountInsideRadius(qp, myTestData.FindData, rBig)
    cSmall += myTestIndex.CountInsideRadius(qp, myTestData.FindData, rSmall)
  if cBig > cSmall and cSmall > 0:
    dim = math.log(cBig/cSmall)/math.log(rBig/rSmall)
  else:
    dim = 0
  print cBig, cSmall, dim
  return ind

                   
# Generate some 2-dimensional data, put it into an index and then
# show the points retrieved.  This is all done as a function of number
# of projections per bucket, number of buckets to use for each index, and
# the number of LSH bucket (the T1 size).  Write out the data so we can
# plot it (in Matlab)
def GraphicalTest(k, l, N):
  numPoints = 1000
  myTestData = TestDataClass(numPoints, 3)  
  ind = index_quad(.1, k, l, N)
  for i in range(0,numPoints):
    ind.InsertIntoTable(i, myTestData.RetrieveData(i))
  i = 42
  r = ind.Find(data[i,:])
  fp = open('lsh_quadtestpoints.txt','w')
  for i in range(0,numPoints):
    if i in r: 
      c = r[i]
    else:
      c = 0
    fp.write("%g %g %d\n" % (data[i,0], data[i,1], c))
  fp.close()
  return r
    

      
def SimpleTest():
  import time
  dim = 250
  numPoints = 10000
  myTestData = RandomTestData()
  myTestData.CreateData(numPoints,dim)
  myTestIndex = index_quad(w=.4, k=10, l=10, N=numPoints)
  startLoad = time.clock()
  for idnum in myTestData.IterateKeys():
    data = myTestData.RetrieveData(idnum)
    myTestIndex.InsertIntoTable(idnum, data)
  endLoad = time.clock()
  print "Time to load %d points is %gs (%gms per point)" % \
    (numPoints, endLoad-startLoad, (endLoad-startLoad)/numPoints*1000.0)

  startRecall = time.clock()
  resCount = 0
  resFound = 0
  for idnum in myTestData.IterateKeys():
    query = myTestData.RetrieveData(idnum)
    res = myTestIndex.Find(query)
    if not res is None and len(res) > 0:
      resFound += 1
    if not res is None:
      resCount += len(res)
  endRecall = time.clock()
  print "Time to recall %d points is %gs (%gms per point" % \
    (numPoints, endRecall-startRecall, (endRecall-startRecall)/numPoints*1000.0)
  print "Found a recall hit all but %d times, average results per query is %g" % \
    (numPoints-resFound, resCount/float(numPoints))
      
      

  
def OutputAllProjections(myTestData, myTestIndex, filename):
  '''Calculate and output all the projected data for an index.'''
  lsh_quadProjector = myTestIndex.projections[0]
  fp = open(filename, 'w')
  for idnum in myTestData.IterateKeys():
    d = myTestData.RetrieveData(idnum)
    (t1, t2, bins, parray) = lsh_quadProjector.CalculateHashes2(d)
    fp.write('%d %d %g %g\n' % (t1, t2, bins[0][0], parray[0][0]))
  fp.close()
  
#   Exact Optimization:
#    For 100000 5-d data use: w=2.91032 and get 0.55401 hits per bin and 0.958216 nn.
#      K=23.3372 L=2.70766 cost is 2.98756
#  Expected statistics for optimal solution:
#    Assuming K=23, L=3
#    p_nn(w) is 0.958216
#    p_any(w) is 0.55401
#    Probability of finding NN for L=1: 0.374677
#    Probability of finding ANY for L=1: 1.26154e-06
#    Probability of finding NN for L=3: 0.75548
#    Probability of finding ANY for L=3: 3.78462e-06
#    Expected number of hits per query: 0.378462

'''
10-D data:
Mean of Python NN data is 0.601529 and std is 0.0840658.
Scaling all distances by 0.788576 for easier probability calcs.
Simple Approximation:
  For 100000 5-d data use: w=4.17052 and get 0.548534 hits per bin and 0.885004 nn.
    K=19.172 L=10.4033 cost is 20.8065
Expected statistics: for simple approximation
  Assuming K=19, L=10
  Probability of finding NN for L=1: 0.0981652
  Probability of finding ANY for L=1: 1.10883e-05
  Probability of finding NN for L=10: 0.644148
  Probability of finding ANY for L=10: 0.000110878
  Expected number of hits per query: 11.0878
Exact Optimization:
  For 100000 5-d data use: w=4.26786 and get 0.556604 hits per bin and 0.887627 nn.
    K=21.4938 L=12.9637 cost is 17.3645
Expected statistics for optimal solution:
  Assuming K=21, L=13
  p_nn(w) is 0.887627
  p_any(w) is 0.556604
  Probability of finding NN for L=1: 0.0818157
  Probability of finding ANY for L=1: 4.53384e-06
  Probability of finding NN for L=13: 0.670323
  Probability of finding ANY for L=13: 5.89383e-05
  Expected number of hits per query: 5.89383
'''

if __name__ == '__main__':
  defaultDims = 10
  defaultW = 2.91032
  defaultK = 10
  defaultL = 1
  defaultClosest = 1000
  defaultMultiprobeRadius = 0
  defaultFileName = 'testData'
  cmdName = sys.argv.pop(0)
  while len(sys.argv) > 0:
    arg = sys.argv.pop(0).lower()
    if arg == '-d':
      arg = sys.argv.pop(0)
      try:
        defaultDims = int(arg)
        defaultFileName = 'testData%03d' % defaultDims
      except:
        print "Couldn't parse new value for defaultDims: %s" % arg
      print 'New default dimensions for test is', defaultDims
    elif arg == '-f':
      defaultFileName = sys.argv.pop(0)
      print 'New file name is', defaultFileName
    elif arg == '-k':
      arg = sys.argv.pop(0)
      try:
        defaultK = int(arg)
      except:
        print "Couldn't parse new value for defaultK: %s" % arg
      print 'New default k for test is', defaultK
    elif arg == '-l':
      arg = sys.argv.pop(0)
      try:
        defaultL = int(arg)
      except:
        print "Couldn't parse new value for defaultL: %s" % arg
      print 'New default l for test is', defaultL
    elif arg == '-c':
      arg = sys.argv.pop(0)
      try:
        defaultClosest = int(arg)
      except:
        print "Couldn't parse new value for defaultClosest: %s" % arg
      print 'New default number closest for test is', defaultClosest
    elif arg == '-w':
      arg = sys.argv.pop(0)
      try:
        defaultW = float(arg)
      except:
        print "Couldn't parse new value for w: %s" % arg
      print 'New default W for test is', defaultW
    elif arg == '-r':
      arg = sys.argv.pop(0)
      try:
        defaultMultiprobeRadius = int(arg)
      except:
        print "Couldn't parse new value for multiprobeRadius: %s" % arg
      print 'New default multiprobeRadius for test is', defaultMultiprobeRadius
    elif arg == '-create':      # Create some uniform random data and find NN
      myTestData = RandomTestData()
      myTestData.CreateData(100000, defaultDims)
      myTestData.SaveData(defaultFileName + '.dat')
      print "Finished creating random data.  Now computing nearest neighbors..."
      myTestData.FindNearestNeighbors(defaultClosest)
      myTestData.SaveNearestNeighbors(defaultFileName + '.nn')
    elif arg == '-histogram':    # Calculate distance histograms
      myTestData = TestDataClass()
      myTestData.LoadData(defaultFileName + '.dat')
      myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
      fp = open(defaultFileName + '.distances', 'w')
      if fp:
        myTestData.ComputeDistanceHistogram(fp)
        fp.close()
      else:
        print "Can't open %s.distances to store NN data" % defaultFileName
    elif arg == '-sanity':
      myTestData = TestDataClass()
      myTestData.LoadData(defaultFileName + '.dat')
      print myTestData.RetrieveData(myTestData.GetRandomQuery())
      print myTestData.RetrieveData(myTestData.GetRandomQuery())
    elif arg == '-b':    # Calculate bucket probabilities
      random.seed(0)
      myTestData = TestDataClass()
      myTestData.LoadData(defaultFileName + '.dat')
      myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
      # ComputePnnPanyCurve(myData, [.291032])
      myTestData.ComputePnnPanyCurve(defaultW)
    elif arg == '-wtest':    # Calculate bucket probabilities as a function of w
      random.seed(0)
      myTestData = TestDataClass()
      myTestData.LoadData(defaultFileName + '.dat')
      myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
      wList = [defaultW*.5**-i for i in range(-10,10)]
      # wList = [defaultW*.5**-i for i in range(-3,3)]
      myTestData.ComputePnnPanyCurve(wList, defaultMultiprobeRadius)
    elif arg == '-ktest':    # Calculate bucket probabilities as a function of k
      random.seed(0)
      myTestData = TestDataClass()
      myTestData.LoadData(defaultFileName + '.dat')
      myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
      # ComputePnnPanyCurve(myData, [.291032])
      kList = [math.floor(math.sqrt(2)**k) for k in range(0,10)]
      kList = [1,2,3,4,5,6,8,10,12,14,16,18,20,22,25,30,35,40]
      myTestData.ComputeKCurve(kList, defaultW, defaultMultiprobeRadius)
    elif arg == '-ltest':    # Calculate bucket probabilities as a function of l
      random.seed(0)
      myTestData = TestDataClass()
      myTestData.LoadData(defaultFileName + '.dat')
      myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
      # ComputePnnPanyCurve(myData, [.291032])
      lList = [math.floor(math.sqrt(2)**k) for k in range(0,10)]
      lList = [1,2,3,4,5,6,8,10,12,14,16,18,20,22,25,30]
      myTestData.ComputeLCurve(lList, w=defaultW, 
        k=defaultK, r=defaultMultiprobeRadius)
    elif arg == '-timing':
      # sys.argv.pop(0)
      timingModels = []
      while len(sys.argv) > 0:
        print "Parsing timing argument", sys.argv[0], len(sys.argv)
        if sys.argv[0].startswith('-'):
          break
        try:
          (w,k,l,r,rest) = sys.argv[0].strip().split(',', 5)
          timingModels.append([float(w), int(k), int(l), int(r)])
        except:
          print "Couldn't parse %s.  Need w,k,l,r" % sys.argv[0]
        sys.argv.pop(0)
      myTestData = TestDataClass()
      myTestData.LoadData(defaultFileName + '.dat')
      myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
      for (w, k, l, r) in timingModels:
        sys.stdout.flush()
        (pnnL1, pnn, panyL1, pany, perQueryTime, numDims) = myTestData.ComputePnnPany(w, k, l, r)
        print "Timing:", w, k, l, r, myTestData.NumPoints(), pnn, pany, perQueryTime*1000.0, numDims
    
    elif arg == '-test':    # Calculate bucket probabilities as a function of l
      random.seed(0)
      myTestData = TestDataClass()
      myTestData.LoadData(defaultFileName + '.dat')
      myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
      # ComputePnnPanyCurve(myData, [.291032])
      myTestData.ComputeLCurve([defaultL], w=defaultW, k=defaultK)
    else:
      print '%s: Unknown test argument %s' % (cmdName, arg)


