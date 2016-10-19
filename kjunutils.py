#-------------------------------------------------------------------------
# KJUN
import gzip
import random, numpy, numpy as np, os, operator, traceback, sys, math, time, ipdb
import numpy.random as ra, ipdb, cPickle as pickle, time, scipy.io as sio
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
    print "Couldn't read pickle file", fName
    traceback.print_exc(file=sys.stderr)

def SavePickle(filename, var, protocol=2):
  try:
    with open(filename, 'wb') as f:
      pickle.dump(var, f, protocol=protocol)
    statinfo = os.stat(filename,)
    if statinfo:
      print "Wrote out", statinfo.st_size, "bytes to", \
        filename
  except:
    print "Couldn't pickle index to file", filename
    traceback.print_exc(file=sys.stderr)

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

def printExpr(expr):
  """ Print the local variables in the caller's frame."""
  from pprint import pprint
  import inspect
  frame = inspect.currentframe()
  try:
    loc = frame.f_back.f_locals
    print expr, '= ', 
    pprint(eval(expr, globals(), loc));
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

def ListOf2dArrayTo3d(mat):
  tmp = np.zeros((len(mat), mat[0].shape[0], mat[0].shape[1]));
  for i in range(len(mat)): tmp[i,:,:] = mat[i]
  return tmp;
