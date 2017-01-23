import sys 
#prefix = '../OptimalLSH/'
prefix = 'OptimalLSH/'
sys.path.insert(0, prefix)
from kjunutils import *
import numpy, time, numpy as np
from numpy import * from numpy.linalg import *
import numpy.random as ra, ipdb, cPickle as pickle, time, scipy.io as sio, numpy.linalg as la
import warnings,sys,os,copy
from choldate import cholupdate, choldowndate

def CalcSqrtBeta(d,t,scale,R,ridge,delta,S_hat):
  return scale*(R * sqrt(d * log((1 + t / (ridge * d)) / delta)) + sqrt(ridge) * S_hat)

def CalcSqrtBetaDet(d,t,scale,R,ridge,delta,S_hat,logdetV):
  return scale*(R * sqrt( logdetV - d*log(ridge) + log (1/(delta**2))  ) + sqrt(ridge) * S_hat)

'''Place holder class that all bandit algorithms must inherit from banditclass '''
class banditclass(object):
    def __init__(self, x0, X, y):
        pass

    def setparams(self, opts):
        pass

    def _process_reward(self, reward, next_index):
        pass

    def _next_arm(self):
        pass

    def _get_reward(self, ind):
        pass

    def runbandit(self, T):
        pass

################################################################################
class LshNonquadWrap(object):
################################################################################
    def __init__(self, lsh, lsh_index_array):
        if (lsh is None):
          return
        self.lsh = lsh
        self.index_array = lsh_index_array

        tmp = numpy.zeros(len(lsh_index_array),dtype=int)
        tmp[self.index_array.tolist()] = range(len(lsh_index_array))
        self.index_array_inv = tmp

    def FindUpto(self, query, max_dist_comp, randomize, invalidList, maxLookup=5000):
        myQuery = np.zeros((len(query),1), dtype=float32)
        myQuery[:,0] = query
        lshInvalidSet = set( [self.index_array_inv[x] for x in invalidList] )
        foundSet, foundListTuple, dbgDict = self.lsh.FindUpto(myQuery, max_dist_comp,
                                          randomize=randomize, invalidSet=lshInvalidSet, maxLookup=maxLookup)
        maxDepth = max([x[1] for x in foundListTuple])
        foundList = [self.index_array[x[0]] for x in foundListTuple]

        #- when there is not enough things, fill it out with
        nFound = len(foundList)
        dbgDict['nRetrievedFromLsh'] = nFound
        N = len(self.index_array)
        if (nFound < max_dist_comp):
          wholeSet = set(xrange(len(self.index_array)))
          #- pick (max_dist_comp - nFound) data points
          remainder = max_dist_comp - nFound
          foundList += ra.permutation(list(wholeSet - set(invalidList) - set(foundList)))[:remainder].tolist()

        return foundList, maxDepth, dbgDict

################################################################################
class LshQuadWrap(object):
################################################################################
    def __init__(self, lsh, lsh_index_array):
        if (lsh is None):
          return
        self.lsh = lsh
        self.index_array = lsh_index_array

        tmp = numpy.zeros(len(lsh_index_array),dtype=int)
        tmp[self.index_array.tolist()] = range(len(lsh_index_array))
        self.index_array_inv = tmp

    def FindUpto(self, query_1_vec, query_2_mat, max_dist_comp, randomize, invalidList, maxLookup=5000):
        d = len(query_1_vec)
        myQuery = np.zeros((d + d**2,1), dtype=float32)
        myQuery[:d,0] = query_1_vec
        myQuery[d:,0] = query_2_mat.ravel()
        lshInvalidSet = set( [self.index_array_inv[x] for x in invalidList] )
        foundSet, foundListTuple, dbgDict = self.lsh.FindUpto(myQuery, max_dist_comp,
                                          randomize=randomize, invalidSet=lshInvalidSet, maxLookup=maxLookup)
        maxDepth = max([x[1] for x in foundListTuple])
        foundList = [self.index_array[x[0]] for x in foundListTuple]

        #- when there is not enough things, fill it out with
        nFound = len(foundList)
        dbgDict['nRetrievedFromLsh'] = nFound
        N = len(self.index_array)
        if (nFound < max_dist_comp):
          wholeSet = set(range(len(self.index_array)))
          #- pick (max_dist_comp - nFound) data points
          remainder = max_dist_comp - nFound
          foundList += ra.permutation(list(wholeSet - set(invalidList) - set(foundList)))[:remainder].tolist()

        return foundList, maxDepth, dbgDict


################################################################################
class oful(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.ridge = None
        self.delta = None
        self.R = None
        self.x0 = x0
        self.sqrt_beta = None
        self.S_hat = None

        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = None
        self.y = y
        self.b = None
        self.X_invVt_norm_sq = None
        self.thetahat = None
        self.shifted = None

        self.invVt = None
        self.validinds = None
        self.est_rewards = None
        self.sqrt_beta_ary = []
        self.debugval = None

    def setparams(self, opts):
        self.ridge = opts['param1']
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.R = opts['R']
        self.S_hat = opts['S_hat']
        self.shifted = opts['shifted']
        self.sqrt_beta_ary = []
        self.t = 1

        X = self.X
        n = self.n
        d = self.d

        thetahat = self.x0
        self.invVt = numpy.eye(d) / self.ridge
        self.validinds = opts['validinds']

        self.b = numpy.zeros(len(thetahat))
        sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)

        self.sqrt_beta = sqrt_beta
        self.sqrt_beta_ary.append(self.sqrt_beta)

        X_invVt_norm_sq = numpy.sum(X * X, axis=1) / self.ridge
        self.X_invVt_norm_sq = X_invVt_norm_sq
        self.est_rewards = numpy.dot(X, thetahat) + sqrt_beta * numpy.sqrt(X_invVt_norm_sq)

    def _next_arm(self):
        validinds = self.validinds
        est_rewards = self.est_rewards

        next_index = validinds[numpy.argmax(est_rewards[validinds])]
        self.debugval[self.t-1] = self.sqrt_beta*numpy.sqrt(self.X_invVt_norm_sq[next_index])
        reward = self._get_reward(next_index)
        self._process_reward(reward, next_index)

        return next_index, reward

    def _process_reward(self, reward, next_index):
        X = self.X
        b = self.b
        X_invVt_norm_sq = self.X_invVt_norm_sq
        invVt = self.invVt
        validinds = self.validinds
        xt = X[next_index, :]
        b += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        X_invVt_norm_sq = X_invVt_norm_sq - (numpy.dot(X, tempval1) ** 2) / (1 + tempval2) # efficient update
        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)             #
        if self.shifted:
            thetahat = numpy.dot(invVt, b) + self.x0
        else:
            thetahat = numpy.dot(invVt, b)
        validinds = numpy.setdiff1d(validinds, next_index)

        self.t += 1
        sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat) #  self.R * numpy.sqrt(self.d * numpy.log((1 + t / self.ridge) / self.delta)) + numpy.sqrt(self.ridge) * S_hat

        self.sqrt_beta = sqrt_beta
        self.sqrt_beta_ary.append(self.sqrt_beta)
        self.X_invVt_norm_sq = X_invVt_norm_sq
        self.invVt = invVt
        self.b = b
        self.thetahat = thetahat
        self.validinds = validinds
        self.est_rewards = numpy.dot(X, thetahat) + sqrt_beta * numpy.sqrt(X_invVt_norm_sq)

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        self.debugval = numpy.zeros(T)
        for i in xrange(T):
            next_index, reward = self._next_arm()
            rewards[i] = reward
            arms[i] = next_index

        return rewards, arms, self.sqrt_beta_ary, self.debugval

################################################################################
class oful_light(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.x0 = x0
        self.X = X
        self.y = y

    def setparams(self, opts):
        self.n, self.d = self.X.shape
        self.ridge = opts['param1']
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.R = opts['R']
        self.S_hat = opts['S_hat']
        self.shifted = opts['shifted']
        self.t = 1
        self.max_dist_comp = opts['max_dist_comp']

        X = self.X
        n = self.n
        d = self.d

        self.do_not_ask = opts['do_not_ask']

        thetahat = self.x0
        self.invVt = numpy.eye(d) / self.ridge
        validinds = setdiff1d(range(n), self.do_not_ask)
#        self.validinds = opts['validinds']
        self.XTy = numpy.zeros(len(thetahat))
        sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)

        self.sqrt_beta = sqrt_beta
#        self.sqrt_beta_ary.append(self.sqrt_beta)

        light_inds = ra.permutation(validinds)[:self.max_dist_comp]
        self.light_inds = light_inds

        subX = X[light_inds,:]
        X_invVt_norm_sq = numpy.sum(subX * subX, axis=1) / self.ridge
        self.expected_rewards = -np.inf*np.ones(n)
        self.expected_rewards[light_inds] = numpy.dot(subX, thetahat) + sqrt_beta * numpy.sqrt(X_invVt_norm_sq)
        self.thetahat = thetahat

    def _next_arm(self):
        next_index = self.light_inds[np.argmax(self.expected_rewards[self.light_inds])]

#         reward = self._get_reward(next_index)
#         self._process_reward(reward, next_index)
#         return next_index, reward
        return next_index

    def _process_reward(self, reward, next_index):
        X = self.X
        XTy = self.XTy
        #X_invVt_norm_sq = self.X_invVt_norm_sq
        invVt = self.invVt
#        validinds = self.validinds

        xt = X[next_index, :]
        XTy += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        #X_invVt_norm_sq = X_invVt_norm_sq - (numpy.dot(X, tempval1) ** 2) / (1 + tempval2)
        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)
        if self.shifted:
            thetahat = numpy.dot(invVt, XTy) + self.x0
        else:
            thetahat = numpy.dot(invVt, XTy)

        # NOTE updating do_not_ask is updated somewhere else
#        self.do_not_ask.append(next_index) # TODO is this done from NEXT side?

        self.t += 1
        sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
        self.sqrt_beta = sqrt_beta
#        self.sqrt_beta_ary.append(self.sqrt_beta)
        self.invVt = invVt
        self.XTy = XTy
        self.thetahat = thetahat

        validinds = setdiff1d(range(self.n),self.do_not_ask)
        light_inds = ra.permutation(validinds)[:self.max_dist_comp]
        #light_inds.sort() # this helps, but let's not do it for now.
        expected_rewards = -np.inf*np.ones(self.n)
        expected_rewards[light_inds] = dot(X[light_inds,:], self.thetahat) + \
            self.sqrt_beta * sqrt(np.sum(X[light_inds,:] * dot(X[light_inds,:], self.invVt),1))

        self.light_inds = light_inds
        self.expected_rewards = expected_rewards

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def _update_do_not_ask(self, next_index):
        self.do_not_ask.append(next_index)

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        for i in xrange(T):
#            next_index, reward = self._next_arm()
            next_index = self._next_arm()

            reward = self.y[next_index]
            self._update_do_not_ask(next_index)

            self._process_reward(reward, next_index)

            rewards[i] = reward
            arms[i] = next_index

        return rewards, arms, self.sqrt_beta_ary, self.debugval


################################################################################
class oful_lazy(banditclass):
################################################################################
    def __init__(self, x0, X, y, subsample_type, C_lazy):
        self.x0 = x0
        self.X = X
        self.y = y
        self.subsample_type = subsample_type
        self.C_lazy = C_lazy

    def setparams(self, opts):
        #- for lsh
        self.maxDepthList = []
        self.nRetrievedList = []
        self.time_lsh_ary = []

        self.n, self.d = self.X.shape
        self.ridge = opts['param1']
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.R = opts['R']
        self.S_hat = opts['S_hat']
        self.shifted = opts['shifted']
        self.t = 1
        self.max_dist_comp = opts['max_dist_comp']
        X = self.X
        n = self.n
        d = self.d

        self.do_not_ask = opts['do_not_ask']

        if (self.subsample_type == 'lsh'):
          self.lsh_max_lookup = opts['lsh_max_lookup']
          assert(opts['lsh'] is not None)
          self.lsh_wrap = LshNonquadWrap(opts['lsh'], opts['lsh_index_array'])
          self.nRetrievedFromLshList = []
#           self.lsh = opts['lsh']
#           self.index_array = opts["lsh_index_array"]
#           tmp = numpy.zeros(self.n)
#           tmp[self.index_array.tolist()] = range(self.n)
#           self.index_array_inv = tmp

        self.thetahat = self.x0
        self.invVt = numpy.eye(d) / self.ridge
        validinds = setdiff1d(range(n), self.do_not_ask)
#        self.validinds = opts['validinds']

        self.XTy = numpy.zeros(len(self.thetahat))
        self.sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
#        self.sqrt_beta_ary.append(self.sqrt_beta)

        self.X_invVt_norm_sq = numpy.sum(X * X, axis=1) / self.ridge

        #- for lazy update.
        self.logdetV = self.d*log(self.ridge)
        self.logdetV_tau = -np.inf
        self.thetatil_tau = np.zeros(self.d)
        self.switch_time_ary = []

        self.expected_rewards = -inf*np.ones(self.n) # this is really a buffer.

        #- further initialization
        self.logdetV_tau = self.logdetV
        expected_rewards, best_index, thetatil, written_inds = self._calc_est_rewards()
        self.thetatil_tau = thetatil
        self.switch_time_ary.append(self.t) # save when the switching happened
        self.written_inds = written_inds

    def _calc_est_rewards(self):
        X = self.X
        expected_rewards = self.expected_rewards
        validinds = setdiff1d(range(self.n), self.do_not_ask)
        if (self.subsample_type == 'no'):
          inv_Vt_norm = np.sqrt(np.sum(X * np.dot(X,self.invVt), 1))
          expected_rewards[:] = numpy.dot(X, self.thetahat) + self.sqrt_beta * inv_Vt_norm
          best_index = validinds[numpy.argmax(expected_rewards[validinds])]
          x = X[best_index,:]
          thetatil = self.thetahat + (self.sqrt_beta / inv_Vt_norm[best_index]) * np.dot(self.invVt,x)
          written_inds = validinds.copy()
        elif (self.subsample_type == 'light' or self.subsample_type == 'lsh'):
          light_inds = ra.permutation(validinds)[0:self.max_dist_comp]
          subX = X[light_inds,:]

          sub_inv_Vt_norm = np.sqrt(np.sum(subX * np.dot(subX,self.invVt), 1))
          expected_rewards[:] = -np.inf
          expected_rewards[light_inds] = numpy.dot(subX, self.thetahat) + self.sqrt_beta * sub_inv_Vt_norm
          best_index_inner = argmax(expected_rewards[light_inds])
          best_index = light_inds[best_index_inner]
          x = X[best_index,:]
          thetatil = self.thetahat + (self.sqrt_beta / sub_inv_Vt_norm[best_index_inner]) * np.dot(self.invVt,x)
          written_inds = light_inds
        else:
          assert False, 'unknown type'
        return expected_rewards, best_index, thetatil, written_inds

    def _calc_est_rewards_lazy(self):
        X = self.X
        expected_rewards = self.expected_rewards
        validinds = setdiff1d(range(self.n), self.do_not_ask)
        if (self.subsample_type == 'no'):
          expected_rewards[:] = numpy.dot(X, self.thetatil_tau)
          best_index = validinds[argmax(expected_rewards[validinds])]
          written_inds = validinds.copy()
        elif (self.subsample_type == 'light'):
          light_inds = ra.permutation(validinds)[0:self.max_dist_comp]
          subX = X[light_inds,:]

          expected_rewards[:] = -np.inf
          expected_rewards[light_inds] = np.dot(subX, self.thetatil_tau)
          best_index = light_inds[ argmax(expected_rewards[light_inds]) ]
          written_inds = light_inds
        elif (self.subsample_type == 'lsh'):
          invalidList = numpy.setdiff1d(range(self.n), validinds)

          #- call lsh
          time_lsh = tic()
          foundList, maxDepth, debugDict = self.lsh_wrap.FindUpto(self.thetatil_tau, self.max_dist_comp, randomize=True, invalidList=invalidList, maxLookup=self.lsh_max_lookup)
          assert( len(numpy.intersect1d(foundList, validinds)) == len(foundList) )
          self.time_lsh_ary.append(toc(time_lsh))
          self.maxDepthList.append(maxDepth)
          self.nRetrievedFromLshList.append(debugDict['nRetrievedFromLsh'])

          #--- compute the inner product
          #- Note that sorting the list improves speed, but I am not doing it currently.
          expected_rewards[:] = -np.inf
          expected_rewards[foundList] = np.dot(self.X[foundList,:], self.thetatil_tau)
          best_index = foundList[numpy.argmax(expected_rewards[foundList])]
          written_inds = foundList
        else:
          assert False, 'unknown type'
        return expected_rewards, best_index, written_inds

    def _next_arm(self):
        next_index = self.written_inds[np.argmax(self.expected_rewards[self.written_inds])]
        return next_index

    def _process_reward(self, reward, next_index):
        X_invVt_norm_sq = self.X_invVt_norm_sq
        invVt = self.invVt
        xt = self.X[next_index, :]

        #- update variables
        self.XTy += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        self.logdetV += log(1 + tempval2)
        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)
        if self.shifted:
            self.thetahat = numpy.dot(invVt, self.XTy) + self.x0
        else:
            self.thetahat = numpy.dot(invVt, self.XTy)

        self.t += 1
        self.sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
#        self.sqrt_beta_ary.append(self.sqrt_beta)
        self.invVt = invVt

        #- prepare for the next arm
        X = self.X

        logdetV_large_enough = self.logdetV > np.log(1.0 + self.C_lazy) + self.logdetV_tau
        if logdetV_large_enough:
          self.logdetV_tau = self.logdetV
#          tic()
          expected_rewards, best_index, thetatil, written_inds = self._calc_est_rewards()
          self.thetatil_tau = thetatil
          self.switch_time_ary.append(self.t) # save when the switching happened
#          print 'switch: ' + str(toc())
        else:
#          tic()
          expected_rewards, best_index, written_inds = self._calc_est_rewards_lazy()
#          print 'no-switch: ' + str(toc())

        self.written_inds = written_inds

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def _update_do_not_ask(self, next_index):
        self.do_not_ask.append(next_index)

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        for i in xrange(T):
            next_index = self._next_arm()

            reward = self.y[next_index]
            self._update_do_not_ask(next_index)

            self._process_reward(reward, next_index)

            rewards[i] = reward
            arms[i] = next_index

        myDict = {}
        myDict['switch_time_ary'] = self.switch_time_ary
        printExpr("myDict['switch_time_ary']",bPretty=False)
        if self.subsample_type == 'lsh':
            myDict['maxDepthList'] = self.maxDepthList
            myDict['time_lsh_ary'] = self.time_lsh_ary
            myDict['nRetrievedFromLshList'] = self.nRetrievedFromLshList
            printExpr("myDict['nRetrievedFromLshList']",bPretty=False)
#            printExpr("myDict['maxDepthList'])",bPretty=False)
#            printExpr("myDict['nRetrievedList']",bPretty=False)
#            printExpr("myDict['time_lsh_ary']",bPretty=False)
        return rewards, arms, [], myDict

################################################################################
class ofulx9(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.ridge = None
        self.delta = None
        self.R = None
        self.x0 = x0
        self.sqrt_beta = None
        self.S_hat = None
        self.shifted = None

        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = None
        self.y = y
        self.b = None
        self.c1 = None
        self.X_invVt_norm_sq = None
        self.thetahat = None
        self.shifted = None

        self.invVt = None
        self.validinds = None
        self.sqrt_beta_ary = []
        self.term1 = None
        self.term2 = None
        self.min_sqrt_eig = None
        self.debugval = None
        self.doEigVal = None

        self.logdetV = None
        self.useSqrtBetaDet = False
        if (self.useSqrtBetaDet):
            print('debug useSqrtBetaDet == True')

    def setparams(self, opts):
        self.ridge = opts['param1']
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.R = opts['R']
        self.S_hat = opts['S_hat']
        self.shifted = opts['shifted']
        self.sqrt_beta_ary = []
        self.t = 1
        self.c1 = opts['c1']

        X = self.X
        n = self.n
        d = self.d
        self.doEigVal = False

        thetahat = self.x0
        self.invVt = numpy.eye(d) / self.ridge
        self.validinds = opts['validinds']
        self.min_sqrt_eig = 1/numpy.sqrt(self.ridge)

        self.b = numpy.zeros(self.d)

        self.logdetV = self.d*log(self.ridge)

        self.X_invVt_norm_sq = numpy.sum(X * X, axis=1) / self.ridge
        if (self.useSqrtBetaDet):
            self.sqrt_beta = CalcSqrtBetaDet(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat,self.logdetV)
        else:
            self.sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
        self.sqrt_beta_ary.append(self.sqrt_beta)
        self.term1 = self.X_invVt_norm_sq
        self.term2 = numpy.dot(X, thetahat)


    def _next_arm(self):
        validinds = self.validinds
        c1 = self.c1
        sqrt_beta = self.sqrt_beta
        term1 = self.term1
        term2 = self.term2
        # greedy_index = validinds[numpy.argmax(term2[validinds])]
        # greedy_value = numpy.sqrt(term1[greedy_index])
        total = term2 + (numpy.sqrt(sqrt_beta) / 4 / c1 /self.min_sqrt_eig) * term1
        next_index = validinds[numpy.argmax(total[validinds])]
        #self.debugval[self.t-1] = numpy.sqrt(sqrt_beta) / 4 / c1 * term1[next_index] / self.min_sqrt_eig

        reward = self._get_reward(next_index)
        self._process_reward(reward, next_index)

        return next_index, reward


    def _process_reward(self, reward, next_index):
        X = self.X
        b = self.b
        X_invVt_norm_sq = self.X_invVt_norm_sq
        invVt = self.invVt
        validinds = self.validinds
        xt = X[next_index, :]
        b += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        self.logdetV += log(1 + tempval2)

        X_invVt_norm_sq = X_invVt_norm_sq - (numpy.dot(X, tempval1) ** 2) / (1 + tempval2) # efficient update
        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)             #
        if self.shifted:
            thetahat = self.x0 + numpy.dot(invVt, b)
        else:
            thetahat = numpy.dot(invVt, b)
        validinds = numpy.setdiff1d(validinds, next_index)

        self.t += 1
#        eig_values = numpy.linalg.eigvalsh(invVt)
#        min_sqrt_eig = numpy.sqrt(numpy.min(eig_values))
        if (self.doEigVal):
          eig_values = numpy.linalg.eigvalsh(invVt)
          min_sqrt_eig = numpy.sqrt(numpy.min(eig_values))
        else:
          min_sqrt_eig = 1/sqrt(self.ridge + self.t)
        self.min_sqrt_eig = min_sqrt_eig

        if (self.useSqrtBetaDet):
            self.sqrt_beta = CalcSqrtBetaDet(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat,self.logdetV)
        else:
            self.sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
        self.sqrt_beta_ary.append(self.sqrt_beta)
        self.term1 = X_invVt_norm_sq
        self.term2 = numpy.dot(X, thetahat)

        self.X_invVt_norm_sq = X_invVt_norm_sq
        self.b = b
        self.invVt = invVt
        self.thetahat = thetahat
        self.validinds = validinds


    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        self.debugval = numpy.zeros(T)
        for i in xrange(T):
            next_index, reward = self._next_arm()
            rewards[i] = reward
            arms[i] = next_index

        return rewards, arms, self.sqrt_beta_ary, self.debugval



################################################################################
class ofulx9_light(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.ridge = None
        self.delta = None
        self.R = None
        self.x0 = x0
        self.sqrt_beta = None
        self.S_hat = None
        self.shifted = None

        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = None
        self.y = y
        self.b = None
        self.c1 = None
        #self.X_invVt_norm_sq = None
        self.thetahat = None
        self.shifted = None

        self.invVt = None
        self.validinds = None
        self.sqrt_beta_ary = []
        self.min_sqrt_eig = None
        self.max_dist_comp = None
        self.debugval = None
        self.doEigVal = None

    def setparams(self, opts):
        self.ridge = opts['param1']
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.R = opts['R']
        self.S_hat = opts['S_hat']
        self.shifted = opts['shifted']
        self.sqrt_beta_ary = []
        self.t = 1
        self.c1 = opts['c1']
        self.max_dist_comp = opts['max_dist_comp']

        X = self.X
        n = self.n
        d = self.d
        self.doEigVal = False

        thetahat = self.x0
        self.invVt = numpy.eye(d) / self.ridge
        self.validinds = opts['validinds']
        self.min_sqrt_eig = 1/numpy.sqrt(self.ridge)

        self.b = numpy.zeros(self.d)

        X_invVt_norm_sq = numpy.sum(X * X, axis=1) / self.ridge
        sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
        self.sqrt_beta = sqrt_beta
        self.sqrt_beta_ary.append(sqrt_beta)
#         self.term1 = self.X_invVt_norm_sq
#         self.term2 = numpy.dot(X, thetahat)
        self.est_rewards = numpy.dot(X, thetahat) + \
            (sqrt(sqrt_beta)/4/self.c1/self.min_sqrt_eig) * X_invVt_norm_sq

        self.light_inds = copy.copy(self.validinds)

    def _next_arm(self):
#        validinds = self.validinds
#        total = term2 + (numpy.sqrt(sqrt_beta) / 4 / c1 /self.min_sqrt_eig) * term1
        light_inds = self.light_inds
        next_index = light_inds[numpy.argmax(self.est_rewards[light_inds])]
        #self.debugval[self.t-1] = numpy.sqrt(sqrt_beta) / 4 / c1 * term1[next_index] / self.min_sqrt_eig

        reward = self._get_reward(next_index)
        self._process_reward(reward, next_index)

        return next_index, reward

    def _process_reward(self, reward, next_index):
        X = self.X
        b = self.b
        #X_invVt_norm_sq = self.X_invVt_norm_sq
        invVt = self.invVt
        validinds = self.validinds
        xt = X[next_index, :]
        b += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        #X_invVt_norm_sq = X_invVt_norm_sq - (numpy.dot(X, tempval1) ** 2) / (1 + tempval2) # efficient update
        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)             #
        if self.shifted:
            thetahat = self.x0 + numpy.dot(invVt, b)
        else:
            thetahat = numpy.dot(invVt, b)
        validinds = numpy.setdiff1d(validinds, next_index)

        self.t += 1
#         eig_values = numpy.linalg.eigvalsh(invVt)
#         min_sqrt_eig = numpy.sqrt(numpy.min(eig_values))
        if (self.doEigVal):
          eig_values = numpy.linalg.eigvalsh(invVt)
          min_sqrt_eig = numpy.sqrt(numpy.min(eig_values))
        else:
          min_sqrt_eig = 1/sqrt(self.ridge + self.t)
        self.min_sqrt_eig = min_sqrt_eig

        sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
        self.sqrt_beta = sqrt_beta
        self.sqrt_beta_ary.append(sqrt_beta)
        #self.term1 = X_invVt_norm_sq
        #self.term2 = numpy.dot(X, thetahat)

        #self.X_invVt_norm_sq = X_invVt_norm_sq
        self.invVt = invVt
        self.b = b
        self.thetahat = thetahat
        self.validinds = validinds

        est_rewards = self.est_rewards
        rand_inds = numpy.random.permutation(validinds)[0:self.max_dist_comp]
        est_rewards[rand_inds] = dot(X[rand_inds,:], self.thetahat) + \
            (sqrt(self.sqrt_beta)/4/self.c1/self.min_sqrt_eig) * \
            np.sum(X[rand_inds,:] * dot(X[rand_inds,:], self.invVt),1)

        self.light_inds = rand_inds
        self.est_rewards = est_rewards

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        self.debugval = numpy.zeros(T)
        for i in xrange(T):
            next_index, reward = self._next_arm()
            rewards[i] = reward
            arms[i] = next_index

        return rewards, arms, self.sqrt_beta_ary, self.debugval


################################################################################
class epsilon_greedy(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.x0 = x0
        self.X = X
        self.y = y


    def setparams(self, opts):
        self.n, self.d = self.X.shape
        self.ridge = opts['param1']
        self.epsilon = opts['param2']

        self.shifted = opts['shifted']
        self.vary_epsilon = opts['vary_epsilon']
        self.t = 1

        X = self.X
        n = self.n
        d = self.d

        self.do_not_ask = opts['do_not_ask']

        thetahat = self.x0
        self.invVt = numpy.eye(d) / self.ridge
        validinds = setdiff1d(range(n), self.do_not_ask)
        self.validinds = validinds
        self.XTy = numpy.zeros(len(thetahat))

        self.expected_rewards = numpy.dot(X, thetahat)
        self.thetahat = thetahat


    def _next_arm(self):
        epsilon = self.epsilon
        if numpy.random.random() > epsilon:
            next_index = self.validinds[np.argmax(self.expected_rewards[self.validinds])]
        else:
            next_index = numpy.random.choice(self.validinds, 1)[0]
        # reward = self._get_reward(next_index)
        #         self._process_reward(reward, next_index)
        #         return next_index, reward
        return next_index


    def _process_reward(self, reward, next_index):
        X = self.X
        XTy = self.XTy
        # X_invVt_norm_sq = self.X_invVt_norm_sq
        invVt = self.invVt
        #        validinds = self.validinds

        xt = X[next_index, :]
        XTy += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)
        if self.shifted:
            thetahat = numpy.dot(invVt, XTy) + self.x0
        else:
            thetahat = numpy.dot(invVt, XTy)

        self.t += 1
        self.invVt = invVt
        self.XTy = XTy
        self.thetahat = thetahat

        if self.vary_epsilon:
            self.epsilon = self.epsilon/self.t

        validinds = setdiff1d(range(self.n), self.do_not_ask)
        self.validinds = validinds

        expected_rewards = numpy.dot(X, thetahat)
        self.expected_rewards = expected_rewards


    def _get_reward(self, ind):
        y = self.y
        return 2 * float(y[ind]) - 1


    def _update_do_not_ask(self, next_index):
        self.do_not_ask.append(next_index)


    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        for i in xrange(T):
            #            next_index, reward = self._next_arm()
            next_index = self._next_arm()

            reward = self.y[next_index]
            self._update_do_not_ask(next_index)

            self._process_reward(reward, next_index)

            rewards[i] = reward
            arms[i] = next_index

        return rewards, arms, self.debugval



################################################################################
class ofulx9_lsh(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.x0 = x0
        self.X = X
        self.y = y

    def setparams(self, opts):
        self.debugDict = dict()
        self.maxDepthList = []
        self.nRetrievedList = []
        self.time_lsh_ary = []

        self.n, self.d = self.X.shape
        self.ridge = opts['param1']
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.R = opts['R']
        self.S_hat = opts['S_hat']
        self.shifted = opts['shifted']
        self.sqrt_beta_ary = []
        self.t = 1
        self.c1 = opts['c1']
        self.max_dist_comp = opts['max_dist_comp']
        self.doEigVal = False

        X = self.X
        n = self.n
        d = self.d

        self.do_not_ask = opts['do_not_ask']

        #- lsh
        self.lsh_max_lookup = opts['lsh_max_lookup']
        assert(opts['lsh'] is not None)
        self.lsh_wrap = LshQuadWrap(opts['lsh'], opts['lsh_index_array'])
        self.nRetrievedFromLshList = []

        thetahat = self.x0
        self.thetahat = thetahat
        self.invVt = numpy.eye(d) / self.ridge
#        self.validinds = opts['validinds']
        self.min_sqrt_eig = 1/numpy.sqrt(self.ridge)
        self.XTy = numpy.zeros(self.d)

        sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
        self.sqrt_beta = sqrt_beta
#        self.sqrt_beta_ary.append(sqrt_beta)

        #-
        self.expected_rewards = -inf*np.ones(self.n)

        #- initial update
        time_lsh = tic()
        validinds = setdiff1d(range(self.n), self.do_not_ask)
        invalidList = self.do_not_ask #numpy.setdiff1d(range(self.n), validinds)
        query_2_mat = self.invVt*(numpy.sqrt(self.sqrt_beta)/4 /self.c1 /self.min_sqrt_eig)
        foundList, maxDepth, debugDict = self.lsh_wrap.FindUpto(self.thetahat, query_2_mat, self.max_dist_comp, randomize=True, invalidList=invalidList, maxLookup=self.lsh_max_lookup)
        assert( len(numpy.intersect1d(foundList, validinds)) == len(foundList) )
        self.time_lsh_ary.append(toc(time_lsh))
        self.maxDepthList.append(maxDepth)
        self.nRetrievedFromLshList.append(debugDict['nRetrievedFromLsh'])

        #--- compute the inner product
        #- Note that sorting the list improves speed, but I am not doing it currently.
        sub_X = self.X[foundList,:]
        term1 = numpy.sum(sub_X * dot(sub_X, self.invVt), axis=1)
        term2 = numpy.dot(sub_X, self.thetahat)
        self.expected_rewards[:] = -np.inf
        self.expected_rewards[foundList] = term2 + (numpy.sqrt(self.sqrt_beta) / 4 / self.c1 / self.min_sqrt_eig)* term1
        self.written_inds = foundList

#    @profile
    def _next_arm(self):
        next_index = self.written_inds[np.argmax(self.expected_rewards[self.written_inds])]
        return next_index

#    @profile
    def _process_reward(self, reward, next_index):
        X = self.X
        XTy = self.XTy

        invVt = self.invVt
        xt = X[next_index, :]
        XTy += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)
        if self.shifted:
            thetahat = self.x0 + numpy.dot(invVt, XTy)
        else:
            thetahat = numpy.dot(invVt, XTy)
        self.t += 1

        if (self.doEigVal):
          eig_values = numpy.linalg.eigvalsh(invVt)
          min_sqrt_eig = numpy.sqrt(numpy.min(eig_values))
        else:
          min_sqrt_eig = 1/sqrt(self.ridge + self.t)
        self.min_sqrt_eig = min_sqrt_eig

        sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
        self.sqrt_beta = sqrt_beta

        self.invVt = invVt
        self.XTy = XTy
        self.thetahat = thetahat

        #- update for the nxt: call lsh
        time_lsh = tic()
        validinds = setdiff1d(range(self.n), self.do_not_ask)
        invalidList = self.do_not_ask #numpy.setdiff1d(range(self.n), validinds)
        query_2_mat = self.invVt*(numpy.sqrt(self.sqrt_beta)/4 /self.c1 /self.min_sqrt_eig)
        foundList, maxDepth, debugDict = self.lsh_wrap.FindUpto(self.thetahat, query_2_mat, self.max_dist_comp, randomize=True, invalidList=invalidList, maxLookup=self.lsh_max_lookup)
        assert( len(numpy.intersect1d(foundList, validinds)) == len(foundList) )
        self.time_lsh_ary.append(toc(time_lsh))
        self.maxDepthList.append(maxDepth)
        self.nRetrievedFromLshList.append(debugDict['nRetrievedFromLsh'])

        #- compute the expected rewards
        sub_X = self.X[foundList,:]
        term1 = numpy.sum(sub_X * dot(sub_X, self.invVt), axis=1)
        term2 = numpy.dot(sub_X, self.thetahat)
        self.expected_rewards[:] = -np.inf
        self.expected_rewards[foundList] = term2 + (numpy.sqrt(self.sqrt_beta) / 4 / self.c1 / self.min_sqrt_eig)* term1
        self.written_inds = foundList

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def _update_do_not_ask(self, next_index):
        self.do_not_ask.append(next_index)

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        for i in xrange(T):
            next_index = self._next_arm()

            reward = self.y[next_index]
            self._update_do_not_ask(next_index)

            self._process_reward(reward, next_index)

            rewards[i] = reward
            arms[i] = next_index


        myDict = {}
        myDict['maxDepthList'] = self.maxDepthList
        myDict['time_lsh_ary'] = self.time_lsh_ary
        myDict['nRetrievedFromLshList'] = self.nRetrievedFromLshList
        printExpr("myDict['maxDepthList']",bPretty=False)
        printExpr("myDict['nRetrievedFromLshList']",bPretty=False)

        return rewards, arms, [], myDict

################################################################################
class thompson(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.delta = None
        self.scale = None
        self.x0 = x0
        self.theta_hat = None
        self.v = None
        self.shifted = None

        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = None

        self.y = y
        self.b = None

        self.theta_til = None

        self.invVt = None
        self.matR = None
        self.validinds = None
        self.sqrt_beta_ary = []
        self.est_rewards = None
        self.debugval = None

    def setparams(self, opts):
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.shifted = opts['shifted']
        self.sqrt_beta_ary = []
        self.t = 1

        X = self.X
        n = self.n
        d = self.d

        # if you look closer, the paper claims that the confidence bound can be
        # tightened by replacing T with t
        self.v = self.scale * numpy.sqrt( 9 * self.d * numpy.log(1/self.delta) )
#         theta_til = self.x0
#         self.theta_til = theta_til
        self.theta_hat = self.x0
#        self.invVt = numpy.eye(d) / self.ridge
        self.invVt = numpy.eye(d)
        self.matR = numpy.eye(d)
        self.validinds = opts['validinds']

        self.b = numpy.zeros(self.d)

#        self.est_rewards = numpy.dot(X, theta_til)

    def _next_arm(self):
        tmp = ra.normal(size=(self.d,))
        theta_til = dot(tmp,self.v*self.matR) + self.theta_hat

        validinds = self.validinds
#        est_rewards = self.est_rewards
        est_rewards = np.dot(self.X, theta_til)

        next_index = validinds[numpy.argmax(est_rewards[validinds])]

        reward = self._get_reward(next_index)
        self._process_reward(reward, next_index)

        return next_index, reward

    def _process_reward(self, reward, next_index):
        X = self.X
        b = self.b
        v = self.v
        invVt = self.invVt
        validinds = self.validinds

        xt = X[next_index, :]
        b += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)
        if self.shifted:
            theta_hat = numpy.dot(invVt, b) + self.x0
        else:
            theta_hat = numpy.dot(invVt, b)

        # update matR this is call by reference.
        choldowndate(self.matR, tempval1 / np.sqrt(1 + tempval2))
#         matR = linalg.cholesky(invVt).T

        validinds = numpy.setdiff1d(validinds, next_index)
        self.t += 1

        self.v = self.scale * numpy.sqrt( 9 * self.d * numpy.log((self.t)/self.delta) )
        self.invVt = invVt
        self.b = b
#        self.theta_til = theta_til
        self.theta_hat = theta_hat
        self.validinds = validinds
#        self.est_rewards = numpy.dot(X, theta_til)

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        for i in xrange(T):
            next_index, reward = self._next_arm()
            rewards[i] = reward
            arms[i] = next_index

        return rewards, arms, self.sqrt_beta_ary, self.debugval


################################################################################
#- for ofulx9_l1
def ofulx9_l1_calc_obj(X, XTy, thHat, invVt, muTil, maxL1Norm):
  iterCntAry = []
  nnzAry = []
  tt = tic()
  fSolAry = np.inf*np.ones(X.shape[0])

  d = len(thHat)
  myData = QuadOptimData(d)
  myData.set_A((2*muTil)*la.inv(invVt))
  term1 = dot(X,invVt)
  term_for_b = -(2*muTil)*XTy
  opt = minFuncQuadL1Options()
  opt['tolX'] = 1e-4
  opt['tolObj'] = 1e-4
  for i in xrange(X.shape[0]):
#   print 'debug..'
#   for i in xrange(201):
    x = X[i,:]
    th0 = thHat + term1[i,:] / (2*muTil)
    th0 = projectOntoL1Ball(th0, maxL1Norm)
    myData.set_b(-x -term_for_b)

    thSol, fSol, info = minFuncQuadL1(myData, maxL1Norm, th0, opt)

    fSolAry[i] = fSol
    iterCntAry.append(info['iterCnt'])
    nnzAry.append((abs(thSol) != 0.0).sum())
    if (i % 1000 == 0 and i != 0):
      minIdx = np.argmin(fSolAry)
      print 'i = %5d, elapsed = %.3f, minIdx = %5d' % (i,toc(tt), minIdx)
      print 'iterCntAry[minIdx] = %5d, nnzAry[minIdx] = %5d, fSolAry[minIdx] = %f' % \
          (iterCntAry[minIdx], nnzAry[minIdx], fSolAry[minIdx])

  dbgDict = {'iterCntAry':iterCntAry, 'nnzAry':nnzAry}
  return fSolAry, dbgDict

################################################################################
class ofulx9_l1(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.ridge = None
        self.delta = None
        self.R = None
        self.x0 = x0
        self.sqrt_beta = None
        self.S_hat = None
        self.shifted = None

        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = None
        self.y = y
        self.XTy = None
        self.c1 = None
        self.X_invVt_norm_sq = None
        self.thetahat = None
        self.shifted = None

        self.invVt = None
        self.validinds = None
        self.sqrt_beta_ary = []
        self.term1 = None
        self.term2 = None
        self.min_sqrt_eig = None
        self.debugval = None
        self.doEigVal = None

        self.logdetV = None
        self.useSqrtBetaDet = False
        if (self.useSqrtBetaDet):
            print('debug useSqrtBetaDet == True')


    def setparams(self, opts):
        self.ridge = opts['param1']
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.R = opts['R']
        self.S_hat = opts['S_hat']
        self.shifted = opts['shifted']
        self.sqrt_beta_ary = []
        self.t = 1
        self.c1 = opts['c1']

        #- for L1
        self.maxL1Norm = opts['maxL1Norm']

        X = self.X
        n = self.n
        d = self.d
        self.doEigVal = False

        thetahat = self.x0
        self.invVt = numpy.eye(d) / self.ridge
        self.validinds = opts['validinds']
        self.min_sqrt_eig = 1/numpy.sqrt(self.ridge)

        self.XTy = numpy.zeros(self.d)

        self.logdetV = self.d*log(self.ridge)

        self.X_invVt_norm_sq = numpy.sum(X * X, axis=1) / self.ridge
        if (self.useSqrtBetaDet):
            self.sqrt_beta = CalcSqrtBetaDet(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat,self.logdetV)
        else:
            self.sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
        self.sqrt_beta_ary.append(self.sqrt_beta)
        self.thetahat = thetahat

    def _next_arm(self):
        validinds = self.validinds

        muTil = self.c1 * (self.sqrt_beta**(-.5)) * self.min_sqrt_eig
        total, dbgDict = ofulx9_l1_calc_obj(self.X, self.XTy, self.thetahat, self.invVt, muTil, self.maxL1Norm)

        next_index = validinds[np.argmin(total[validinds])]
        print 'self.t = %3d, next_index = %6d\n' % (self.t,next_index)

        reward = self._get_reward(next_index)
        self._process_reward(reward, next_index)

        return next_index, reward

    def _process_reward(self, reward, next_index):
        X = self.X
        XTy = self.XTy
        X_invVt_norm_sq = self.X_invVt_norm_sq
        invVt = self.invVt
        validinds = self.validinds
        xt = X[next_index, :]
        XTy += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        self.logdetV += log(1 + tempval2)

        X_invVt_norm_sq = X_invVt_norm_sq - (numpy.dot(X, tempval1) ** 2) / (1 + tempval2) # efficient update
        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)             #
        if self.shifted:
            thetahat = self.x0 + numpy.dot(invVt, XTy)
        else:
            thetahat = numpy.dot(invVt, XTy)
        validinds = numpy.setdiff1d(validinds, next_index)

        self.t += 1
        if (self.doEigVal):
          eig_values = numpy.linalg.eigvalsh(invVt)
          min_sqrt_eig = numpy.sqrt(numpy.min(eig_values))
        else:
          min_sqrt_eig = 1/sqrt(self.ridge + self.t)
        self.min_sqrt_eig = min_sqrt_eig

        if (self.useSqrtBetaDet):
            self.sqrt_beta = CalcSqrtBetaDet(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat,self.logdetV)
        else:
            self.sqrt_beta = CalcSqrtBeta(self.d,self.t,self.scale,self.R,self.ridge,self.delta,self.S_hat)
        self.sqrt_beta_ary.append(self.sqrt_beta)

        self.X_invVt_norm_sq = X_invVt_norm_sq
        self.XTy = XTy
        self.invVt = invVt
        self.thetahat = thetahat
        self.validinds = validinds

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        self.debugval = numpy.zeros(T)
        for i in xrange(T):
            next_index, reward = self._next_arm()
            rewards[i] = reward
            arms[i] = next_index

        return rewards, arms, self.sqrt_beta_ary, self.debugval

################################################################################
class thompson_light(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.delta = None
        self.scale = None
        self.x0 = x0
        self.theta_hat = None
        self.v = None
        self.shifted = None

        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = None

        self.y = y
        self.b = None

        self.theta_til = None

        self.invVt = None
        self.matR = None
        self.validinds = None
        self.light_inds = None
        self.sqrt_beta_ary = []
        self.est_rewards = None
        self.max_dist_comp = None
        self.debugval = None

    def setparams(self, opts):
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.shifted = opts['shifted']
        self.sqrt_beta_ary = []
        self.t = 1
        self.max_dist_comp = opts['max_dist_comp']

        X = self.X
        n = self.n
        d = self.d

        # if you look closer, the paper claims that the confidence bound can be
        # tightened by replacing T with t
        self.v = self.scale * numpy.sqrt( 9 * self.d * numpy.log(1/self.delta) )
#         theta_til = self.x0
#         self.theta_til = theta_til
        self.theta_hat = self.x0
#        self.invVt = numpy.eye(d) / self.ridge
        self.invVt = numpy.eye(d)
        self.matR = numpy.eye(d)
        self.validinds = opts['validinds']

        self.b = numpy.zeros(self.d)
        self.light_inds = self.validinds

        self.est_rewards = np.zeros(self.n)

    def _next_arm(self):
        tmp = ra.normal(size=(self.d,))
        theta_til = dot(tmp,self.v*self.matR) + self.theta_hat

        light_inds = self.light_inds
        self.est_rewards[light_inds] = dot(self.X[light_inds,:], theta_til)

        next_index = light_inds[numpy.argmax(self.est_rewards[light_inds])]

        reward = self._get_reward(next_index)
        self._process_reward(reward, next_index)

        return next_index, reward

    def _process_reward(self, reward, next_index):
        X = self.X
        b = self.b
        v = self.v
        invVt = self.invVt
        validinds = self.validinds

        xt = X[next_index, :]
        b += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)
        if self.shifted:
            theta_hat = numpy.dot(invVt, b) + self.x0
        else:
            theta_hat = numpy.dot(invVt, b)

        # update matR this is call by reference.
        choldowndate(self.matR, tempval1 / np.sqrt(1 + tempval2))
#         tmp = ra.normal(size=(self.d,))
#         theta_til = dot(tmp,v*self.matR) + theta_hat

#         matR = linalg.cholesky(invVt).T
#         tmp = ra.normal(size=(self.d,))
#         theta_til = dot(tmp,v*matR) + theta_hat

        validinds = numpy.setdiff1d(validinds, next_index)
        self.t += 1

        self.v = self.scale * numpy.sqrt( 9 * self.d * numpy.log((self.t)/self.delta) )
        self.invVt = invVt
        self.b = b
#        self.theta_til = theta_til
        self.theta_hat = theta_hat
        self.validinds = validinds
#        self.est_rewards = numpy.dot(X, theta_til)

        light_inds = numpy.random.permutation(validinds)[0:self.max_dist_comp]
#        self.est_rewards[light_inds] = dot(X[light_inds,:], theta_til)
       # dot(X[light_inds,:], self.thetahat) + \
       #     self.sqrt_beta * sqrt(np.sum(X[light_inds,:] * dot(X[light_inds,:], self.invVt),1))

        self.light_inds = light_inds

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        for i in xrange(T):
            next_index, reward = self._next_arm()
            rewards[i] = reward
            arms[i] = next_index

        return rewards, arms, self.sqrt_beta_ary, self.debugval

################################################################################
class thompson_lsh(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.x0 = x0
        self.X = X
        self.y = y

    def setparams(self, opts):
        #- for lsh
        self.debugDict = dict()
        self.maxDepthList = []
        self.nRetrievedList = []
        self.time_lsh_ary = []

        self.n, self.d = self.X.shape
        self.scale = opts['param2']
        self.delta = opts['delta']
        self.shifted = opts['shifted']
        self.t = 1
        self.max_dist_comp = opts['max_dist_comp']

#        self.do_chol_onestep = opts['do_chol_onestep']

        #- on lsh
        self.lsh_max_lookup = opts['lsh_max_lookup']
        assert(opts['lsh'] is not None)
        self.lsh_wrap = LshNonquadWrap(opts['lsh'], opts['lsh_index_array'])
        self.nRetrievedFromLshList = []

        X = self.X
        n = self.n
        d = self.d

        self.do_not_ask = opts['do_not_ask']

        self.v = self.scale * numpy.sqrt( 9 * self.d * numpy.log(1/self.delta) )
        self.thetahat = self.x0
        self.invVt = numpy.eye(d) # I guess ridge parameter is 1 for TS
#        if (self.do_chol_onestep):
#           self.matR = numpy.eye(d)
#           matR = self.matR
#         else:
#           matR = np.eye(d)
        matR = np.eye(d)

        self.XTy = numpy.zeros(self.d)
        self.expected_rewards = -np.inf*np.ones(self.n)

        #--- prepare for the next arm
        #- perform sampling
        tmp = ra.normal(size=(self.d,))
        theta_til = dot(tmp,self.v*matR) + self.thetahat

        #- call lsh
        validinds = setdiff1d(range(self.n), self.do_not_ask)
        invalidList = self.do_not_ask #numpy.setdiff1d(range(self.n), validinds)
        time_lsh = tic()
        foundList, maxDepth, debugDict = self.lsh_wrap.FindUpto(theta_til, self.max_dist_comp, randomize=True, invalidList=invalidList, maxLookup=self.lsh_max_lookup)
        assert( len(numpy.intersect1d(foundList, validinds)) == len(foundList) )
        self.time_lsh_ary.append(toc(time_lsh))
        self.maxDepthList.append(maxDepth)
        self.nRetrievedFromLshList.append(debugDict['nRetrievedFromLsh'])
        self.expected_rewards[:] = -np.inf
        self.expected_rewards[foundList] = np.dot(self.X[foundList,:], theta_til)
        self.written_inds = foundList

    def _next_arm(self):
        next_index = self.written_inds[np.argmax(self.expected_rewards[self.written_inds])]
        return next_index

    def _process_reward(self, reward, next_index):
        X = self.X
        XTy = self.XTy
        v = self.v
        invVt = self.invVt

        xt = X[next_index, :]
        XTy += reward * xt
        tempval1 = numpy.dot(invVt, xt)
        tempval2 = numpy.dot(tempval1, xt)

        invVt = invVt - numpy.outer(tempval1, tempval1) / (1 + tempval2)
        if self.shifted:
            thetahat = numpy.dot(invVt, XTy) + self.x0
        else:
            thetahat = numpy.dot(invVt, XTy)

        #- update matR this is call by reference.
        #- note that the second argument will be corrupted, so be sure to do .copy()
#         if self.do_chol_onestep:
#           choldowndate(self.matR, tempval1 / np.sqrt(1 + tempval2))
#           matR = self.matR
#         else:
#           #         matR = linalg.cholesky(invVt).T
#           matR = la.cholesky(invVt).T
        matR = la.cholesky(invVt).T

        self.t += 1

        self.v = self.scale * numpy.sqrt( 9 * self.d * numpy.log((self.t)/self.delta) )
        self.invVt = invVt
        self.XTy = XTy
        self.thetahat = thetahat

        #--- prepare for the next arm
        #- perform sampling
        tmp = ra.normal(size=(self.d,))
        theta_til = dot(tmp,self.v*matR) + self.thetahat

        #- call lsh
        validinds = setdiff1d(range(self.n), self.do_not_ask)
        invalidList = self.do_not_ask
        time_lsh = tic()
        foundList, maxDepth, debugDict = self.lsh_wrap.FindUpto(theta_til, self.max_dist_comp, randomize=True, invalidList=invalidList, maxLookup=self.lsh_max_lookup)
        assert( len(numpy.intersect1d(foundList, validinds)) == len(foundList) )
        self.time_lsh_ary.append(toc(time_lsh))
        self.maxDepthList.append(maxDepth)
        self.nRetrievedFromLshList.append(debugDict['nRetrievedFromLsh'])
        self.expected_rewards[:] = -np.inf
        self.expected_rewards[foundList] = np.dot(self.X[foundList,:], theta_til)
        self.written_inds = foundList

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def _update_do_not_ask(self, next_index):
        self.do_not_ask.append(next_index)

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T, dtype=int)
        for i in xrange(T):
            next_index = self._next_arm()

            reward = self.y[next_index]
            self._update_do_not_ask(next_index)

            self._process_reward(reward, next_index)

            rewards[i] = reward
            arms[i] = next_index

        myDict = {}
        myDict['maxDepthList'] = self.maxDepthList
        myDict['time_lsh_ary'] = self.time_lsh_ary
        myDict['nRetrievedFromLshList'] = self.nRetrievedFromLshList
        printExpr("myDict['maxDepthList']",bPretty=False)
        printExpr("myDict['nRetrievedFromLshList']",bPretty=False)

        return rewards, arms, [], myDict

################################################################################
class nearest_neighbor(banditclass):
################################################################################
    def __init__(self, x0, X, y):
        self.x0 = x0
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = None
        self.est_rewards = None
        self.y = y
        self.validinds = None

    def setparams(self, opts):
        self.t = 1
        self.est_rewards = numpy.dot(self.X, self.x0)
        self.validinds = opts['validinds']

    def _next_arm(self):
        validinds = self.validinds
        est_rewards = self.est_rewards

        next_index = validinds[numpy.argmax(est_rewards[validinds])]

        reward = self._get_reward(next_index)
        self._process_reward(reward, next_index)

        return next_index, reward

    def _process_reward(self, reward, next_index):
        validinds = self.validinds
        validinds = numpy.setdiff1d(validinds, next_index)
        t = self.t
        t += 1
        self.t = t
        self.validinds = validinds

    def _get_reward(self, ind):
        y = self.y
        return 2*float(y[ind])-1

    def runbandit(self, T):
        rewards = numpy.zeros(T)
        arms = numpy.zeros(T)
        for i in xrange(T):
            next_index, reward = self._next_arm()
            rewards[i] = reward
            arms[i] = next_index

        return rewards, arms, None, None

################################################################################
# bandit interface
################################################################################

# banditClassDict = {
#     'oful_light': oful_light,
#     'ofulx9_lsh': ofulx9_lsh,
#     'ts_lsh':     thompson_lsh,
#     'oful_lazy_lsh': oful_lazy,
# }

def bandit_init_options():
  """
  A bunch of parameters lumped together. Not quite ideal, but we don't have time.
  """
  opts = {'param1':         1.,               #ridge
          'param2':         0.0001,            #scale
          'delta':          0.1,
          'R':              1.,
          'S_hat':          1.,
          'shifted':        True,
          'max_dist_comp':  500,
          'c1':             4.0,
          'lsh_max_lookup': 5000,
          'lsh':            None,
          'lsh_index_array': None,
          'lazy_subsample_type':  'lsh',
          'lazy_C':         10**.5,
#          'do_chol_onestep':True,
  }
  return opts

def bandit_init(algo_name, x0Idx, X, opts):
  #- default options
  opts['do_not_ask'] = [x0Idx]

  if (algo_name == 'oful_light'):
    banditObj = oful_light(X[x0Idx,:], X, None)
  elif (algo_name == 'ofulx9_lsh'):
    banditObj = ofulx9_lsh(X[x0Idx,:], X, None)
  elif algo_name == 'ts_lsh':
    banditObj = thompson_lsh(X[x0Idx,:], X, None)
  elif algo_name == 'oful_lazy_lsh':
    banditObj = oful_lazy(X[x0Idx,:], X, None, 'lsh', opts['lazy_C'])
  elif algo_name == 'epsilon_greedy':
    banditObj = epsilon_greedy(X[x0Idx,:], X, None)
  else:
    assert False, 'algo_name not recognized: %s' % algo_name

  banditObj.setparams(opts)

  context = banditObj.__dict__
  context['algo_name'] = algo_name
  # do not save exptRewards
  del(context['X'])
  del(context['y'])

  if (algo_name in ['oful_lazy_lsh','ofulx9_lsh', 'ts_lsh']):
    lw = context['lsh_wrap']
    lw.lsh = None
    context['lsh_wrap'] = lw.__dict__

  bandit_add_prefixes(context)
  return context

def bandit_add_prefixes(context):
  #- append prefix '_bo_'
  for k in context.keys():
    context['_bo_'+k] = context.pop(k)

def bandit_extract_context(biggerDict):
  retDict = {}
  for (k,v) in biggerDict.iteritems():
    if k.startswith('_bo_'):
      retDict[k] = v
  return retDict

def bandit_next_arm(context):
  #- find the best guy.
  expected_rewards = context['_bo_expected_rewards']
  expected_rewards[context['_bo_do_not_ask']] = -np.inf
  next_index = np.argmax(expected_rewards)
  #- update do_not_ask
  context['_bo_do_not_ask'].append(next_index)
  return next_index

#- argDict must contain 'lsh'
def bandit_update(context, X, pulled_arm, reward, argDict={}):
  for k in context.keys():
    context[k[4:]] = context.pop(k)

  algo_name = context['algo_name']
  if (algo_name == 'oful_light'):
    banditObj = oful_light(None, None, None)
  elif (algo_name == 'ofulx9_lsh'):
    banditObj = ofulx9_lsh(None,None,None)
    lsh_wrap_obj = LshQuadWrap(None,None)
  elif algo_name == 'ts_lsh':
    banditObj = thompson_lsh(None,None,None)
    lsh_wrap_obj = LshNonquadWrap(None,None)
  elif algo_name == 'oful_lazy_lsh':
    banditObj = oful_lazy(None,None,None,None,None)
    lsh_wrap_obj = LshNonquadWrap(None,None)
  elif algo_name == 'epsilon_greedy':
      banditObj = epsilon_greedy(None,None,None)
  else:
    assert False, 'algo_name not recognized: %s' % algo_name

  if (algo_name in ['oful_lazy_lsh','ofulx9_lsh', 'ts_lsh']):
    lw_dict = context['lsh_wrap']
    lw_dict['lsh'] = argDict['lsh']
    context['lsh_wrap'] = lsh_wrap_obj
    context['lsh_wrap'].__dict__ = lw_dict

  #- recover the bandit object
  banditObj.__dict__ = context
  banditObj.X = X
  banditObj.invVt = np.array(banditObj.invVt, dtype=float64)
#   if (algo_name == 'ts_lsh' and banditObj.do_chol_onestep):
#     banditObj.matR = np.array(banditObj.matR, dtype=float64)
  banditObj.thetahat = np.array(banditObj.thetahat, dtype=float64)
  banditObj.XTy = np.array(banditObj.XTy, dtype=float64)
  banditObj.expected_rewards = np.array(banditObj.expected_rewards, dtype=float64)

  #- process reward!!
  banditObj._process_reward(reward, pulled_arm)

  #- remove unnecessary stuff.
  del(banditObj.__dict__['X'])
  if (algo_name in ['oful_lazy_lsh','ofulx9_lsh', 'ts_lsh']):
    lw = context['lsh_wrap']
    lw.lsh = None
    context['lsh_wrap'] = lw.__dict__

  assert banditObj.__dict__ is context # just to really ensure

  bandit_add_prefixes(context)



