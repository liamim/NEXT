#-------------------------------------------------------------------------
# KJUN
from numpy import *
import gzip, random, numpy, numpy as np, os, operator, traceback, sys, math, time, ipdb
import numpy.random as ra, ipdb, cPickle as pickle, time, scipy.io as sio, numpy.linalg as la;
import itertools          # For Multiprobe
import copy
from datetime import datetime
#from distcolors import get_distinguishable_colors

################################################################################
# Pickle
################################################################################
def LoadPickle(fName):
  """ load a pickle file. Assumes that it has one dictionary object that points to
 many other variables."""
  if type(fName) == str:
    try:
      fp = open(fName, 'rb')
    except:
      print "Couldn't open %s" % (fName)
      return None
  else:
    fp = fName
  try:
    ind = pickle.load(fp)
    fp.close()
    return ind
  except:
    print "Couldn't read the pickle file", fName
    traceback.print_exc(file=sys.stderr)
    ipdb.set_trace();

def SavePickle(filename, var, protocol=2):
  try:
    with open(filename, 'wb') as f:
      pickle.dump(var, f, protocol=protocol)
    statinfo = os.stat(filename,)
    if statinfo:
      print "Wrote out", statinfo.st_size, "bytes to", \
        filename
  except:
    print "Couldn't pickle the file", filename
    traceback.print_exc(file=sys.stderr)
    ipdb.set_trace();

def LoadPickleGzip(fName):
  if type(fName) == str:
    try:
      fp = gzip.GzipFile(fName, 'rb')
    except:
      print "Couldn't open %s" % (fName)
      return None
  else:
    fp = fName
  try:
    ind = pickle.load(fp)
    fp.close()
    return ind
  except:
    print "Couldn't read pickle file", fName
    traceback.print_exc(file=sys.stderr)

def SavePickleGzip(filename, var, protocol=2):
  try:
    with gzip.GzipFile(filename, 'wb') as f:
      pickle.dump(var, f, protocol=protocol)
    statinfo = os.stat(filename,)
    if statinfo:
      print "Wrote out", statinfo.st_size, "bytes to", \
        filename
  except:
    print "Couldn't pickle index to file", filename
    traceback.print_exc(file=sys.stderr)


# saves variables in 'varList' to filename, and it searches variables from given 'dic'
# typically, if you want to save variables from current python context
# set 'locals()' as parameter 'dic'
def savePickleFromDic(varList, fileName, dic):
  varDic = {};
  for k in varList:
    varDic[k] = dic[k];
  f = open(fileName, 'wb');
  pickle.dump(varDic, f);
  f.close();


def tic():
    """
    equivalent to Matlab's tic. It start measuring time.
    returns handle of the time start point.
    """
    global gStartTime
    gStartTime = datetime.utcnow();
    return gStartTime

def toc(prev=None):
    """
    get a timestamp in seconds. Time interval is from previous call of tic() to current call of toc().
    You can optionally specify the handle of the time ending point.
    """
    if prev==None: prev = gStartTime;
    return (datetime.utcnow() - prev).total_seconds();

def printExpr(expr, bPretty=True):
  """ Print the local variables in the caller's frame."""
  from pprint import pprint
  import inspect
  frame = inspect.currentframe()
  try:
    loc = frame.f_back.f_locals
    print expr, '= ', 
    if (bPretty):
      pprint(eval(expr, globals(), loc));
    else:
      print(eval(expr, globals(), loc)); 
  finally:
    del frame

def readdata(path):
    # features = sio.loadmat(path + 'features_allshoes_8_normalized.mat')
    features = sio.loadmat(path + 'X_d1000_layer7.pkl')
    labels = sio.loadmat(path + 'Labels.mat')
    colorlabels = sio.loadmat(path + 'ColorLabel_new.mat')
    return features['features_all'], labels['Labels'], colorlabels['RedLabel']

# saves variables in 'varList' to filename, and it searches variables from given 'dic'
# typically, if you want to save variables from current python context
# set 'locals()' as parameter 'dic'
def SavePickleFromDic(fileName, dic, varList):
  varDic = {};
  for k in varList:
    varDic[k] = dic[k];
  f = open(fileName, 'wb');
  pickle.dump(varDic, f);
  f.close();


def SaveToDict(dic, varList):
  varDic = {};
  for k in varList:
    varDic[k] = dic[k];
  return varDic;

def ListOf2dArrayTo3d(mat):
  tmp = np.zeros((len(mat), mat[0].shape[0], mat[0].shape[1]));
  for i in range(len(mat)): tmp[i,:,:] = mat[i]
  return tmp;


################################################################################
# Quadratic optimization
################################################################################

def projectOntoL1Ball(v, b):
  """
  PROJECTONTOL1BALL Projects point onto L1 ball of specified radius.

  w = ProjectOntoL1Ball(v, b) returns the vector w which is the solution
    to the following constrained minimization problem:

     min   ||w - v||_2
     s.t.  ||w||_1 <= b.

    That is, performs Euclidean projection of v to the 1-norm ball of radius
    b.

  Author: John Duchi (jduchi@cs.berkeley.edu)
  """
  assert b >= 0, 'Radius of L1 ball is negative';
  assert v.dtype == np.float64;
  if (np.linalg.norm(v,1) < b):
    return v;
  u = np.abs(v);    # this makes a copy
  u[::-1].sort();   # reverse sorting (in-place sort)
  sv = np.cumsum(u);
  rho = np.where(u > ((sv-b) / np.arange(1,len(u)+1,dtype=np.float64)))[0][-1];
  th = np.maximum(0.0, (sv[rho] - b) / (rho+1));
  w = np.sign(v) * np.maximum(abs(v) - th, 0.0);
  return w;

class QuadOptimData(object):
  def __init__(self, dim):
    self.dim = dim;
    self.A = np.zeros((dim,dim));
    self.b = np.zeros(dim);

  def set_A(self, A):
    self.A[:,:] = A;

  def set_b(self, b):
    self.b[:] = b;

def objQuad(th, data, opt):
  f = None;
  g = None;
  v = dot(data.A,th);
  if (opt == 1 or opt == 3):
    f = .5*dot(th,v) + dot(th,data.b); #data.x) - dot(data.thHat,v);
  if (opt == 2 or opt == 3):
    g = v + data.b;
  return f,g;

def minFuncQuadL1Options():
  ret = dict();
  ret['debug'] = False;
  ret['maxIter'] = 400;
  ret['maxLineSearch'] = 10;
  ret['tolX'] = 1e-7;
  ret['tolObj'] = 1e-7;
  ret['alpha0'] = .01; # 2.0/maxEigVal(data.A) is a good heuristic.
  ret['line_c'] = 1e-3; # line_* is for line search
  ret['line_tau'] = .1;
  ret['line_tau0'] = 1.5; # factor to multiply before line search 

  return ret;

def minFuncQuadL1(qoData, maxL1Norm, th0, opt):
  myObj = lambda th, opt: objQuad(th, qoData, opt); 
  
  debug = opt['debug'];
  th = th0;
  if (debug):
    nLineSearchAry = [];
    objValAry = [];
    alphaAry = [];
  maxLineSearch = opt['maxLineSearch'];
  tolX = opt['tolX'];
  tolObj = opt['tolObj'];
  alphaOld = opt['alpha0'];
  line_c = opt['line_c'];
  line_tau = opt['line_tau'];
  line_tau0 = opt['line_tau0'];
  fCnt = 0;
  gCnt = 0;
  bConvergedFVal = False;
  bConvergedX = False;
  #tic();
  f, trash = myObj(th,1);
  for k in range(opt['maxIter']):
    trash, g = myObj(th,3); 
    p = -g;  
#    p = -np.dot(qoData.invA, g);  
    gCnt += 1;

    #- perform backtracking
    alpha = line_tau0 * alphaOld;
    thCur = th;
    for cnt in range(1,maxLineSearch+1):
#     cnt = 1;
#     while cnt <= maxLineSearch:
      thNew = projectOntoL1Ball(thCur + alpha*p, maxL1Norm);
      fNew, trash = myObj(thNew, 1);
      fCnt += 1;
      if (fNew < f - alpha*line_c):
        break;
      elif la.norm(thNew - thCur) / (1+la.norm(thCur)) < tolX:
        if (fNew > f):
          thNew = thCur;
          fNew = f;
        break;
      alpha *= line_tau;
#      cnt += 1;

    if (debug):
      nLineSearchAry.append(cnt);
      objValAry.append(f);
      alphaAry.append(alpha);

    if (abs(f - fNew) / (1+abs(f)) <= tolObj):
      bConvergedFVal = True;
      break;
    elif la.norm(thNew - th) / (1 + la.norm(th)) < tolX: 
      bConvergedX = True;
      break;

    th = thNew;
    f = fNew;
    alphaOld = alpha;
  #- thNew: the solution

  #printExpr('toc()');
  iterCnt = k + 1;

  info = {'iterCnt':iterCnt, 'fCnt':fCnt, 'gCnt':gCnt, 'bConverged': bConvergedFVal or bConvergedX};
  
  debugDict = None;
  if (debug):
    info['debugDict'] = {'nLineSearchAry':nLineSearchAry, 'objValAry':objValAry, 'alphaAry':alphaAry }; 
    return thNew, fNew, info;
  else:
    return thNew, fNew, info;

#--------- For OFUL.

