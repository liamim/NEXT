"""
* make mapping, targets to features
* work with Zappos dataset
2. generalize to n users, not 1 user
3. choose initial sampling arm
    * myApp.py getQuery/processAnswer help this
    * V, b, theta_hat need to be stored per user
    * add new key to butler.particpants[i]
* make launching easier

## 2016-05-17
### Features
* Download features from internet, assume images have been uploaded (it takes
about 3 hours to do via S3 from the university
* do_not_ask is implemented

### Bottlenecks
* argmax is really slow; it uses a for loop in Python. I'll look into using
    NumPy to speed this up. Extrapolating to 50k features, it will take about
    25 minutes to answer one question

| Trial | App.py:getQuery | myApp.py:getQuery | get_X * 2   | argmax | Total |
| load  | 2.27            | 0.24              | 1.93 (*1)   | 0      | 2.17  |
| q0    | 10.38           | 1.02              | 1.96 + 2.27 | 0      | 5.25  |
| q1    | 9.3             | 1.08              | 1.97 + 1.95 | 4.49   | 9.49  |
| q2    | 14.69           | 1.15              | 3.83 + 1.97 | 7.25   | 14.2  |
(run on 2016-05-18 10:00 on c3.large machine with 2k shoes)

| Trial | App.py:getQuery | myApp.py:getQuery | get_X * 2   | argmax | Total |
| load  | 2.28            | 0.13              | 2.0*1       | 0      | 2.38  |
| q0    | 9.69            | 1.28              | 2.03 + 2.04 | 4.54   | 9.89  |
(run on 2016-05-18 11:00 on r3.large machine with 2k shoes)

| Trial | App.py:getQuery | myApp.py:getQuery | get_X * 2   | argmax | Total |
| load  | 51.57           | 0.58              | 50.8*1      | -      | 51.38 |
| q0    | 115             | 1.11              | 50.5*1      | 58.7   |       |
| q1    | 116             | 1.24              | 51.1 + 50.8 | 63     |       |
(run on 2016-05-11:30 on r3.large machine with 50k shoes)

inverting V does not take the most time in argmax_reward
time to invert (1000, 1000) matrix = 0.0263409614563

*summary:* from this, we need to (a) optimize getting X and (b) optimize
argmax_reward

After speeding up get_X to on the order of 0.2secs:

| Trial | App.py getQuery |
| load | 0.36 secs |
| q0 | not written down|
| q1 | get_X: 0.144 + 0.02, time_to_invert: 2.2secs, argmax_reward = 65secs

So most of the time is spent in OFUL:argmax

After speeding up argmax:

| Function            | Time 0   | Time 1 | Time 2 | Time 3 |
| ------------------- | -----    | -----  | ------ |        |
| myApp:processAnswer | 1.00     | 1.1    | 0.4    | 0.33   |
| alg:processAnswer   | 2.35     | 1.2    | 1.50   | 1.14   |
| App:processAnswer   | 3.38     | 2.3    | 1.9    |        |
| myApp:getQuery      | 1.29     | 1.43   | 0.73   | 0.497  |
| alg:getQuery        | 0.49     | 0.39   | 1.56   | 0.45   |
| App:getQuery        | 3.42     | 3.2    | 2.7    |        |
| ------------------- | -------- |        |        |        |
| Total               | 6.8      |        |        |        |

0. speeds up calculating V^{-1}
1. speeds up arg max
2.5 Moves from 2k to 50k shoes?
2. speeds up storing long lists in database
3. merge Kevin's changes from #101 in.

| Task                  | Time |
| load                  | 1.7  |
| choosing initial shoe | 6.5  |
| q1 yes/no             | 3.6  |
| q2 yes/no             | 3.6  |
| q3 yes/no             | 3.8  |

alg:processAnswer bottleneck: PermStore:set_many, MongoDB:update_one
alg:getQuery 0.5s in PermStore:get, MongoDB:find_one
"""

from __future__ import division
import numpy as np
import next.utils as utils
import time
import cPickle as pickle
import os
import json
#import kjunutils
#prefix = '/Users/aniruddha/Dropbox/2016/NEXT_v1/apps/ImageSearch/algs/OFUL_Hashing/'
#import sys
#sys.path.insert(0, prefix)
from next.lib.hash import lsh_kjun_v3 as hash
#import lsh_kjun
#import pdb


# TODO: change this to 1
reward_coeff = 1.00

def timeit(fn_name=''):
    def timeit_(func, *args, **kwargs):
        def timing(*args, **kwargs):
            start = time.time()
            r = func(*args, **kwargs)
            utils.debug_print('')
            utils.debug_print("function {} took {} seconds".format(fn_name, time.time() - start))
            utils.debug_print('')
            return r
        return timing
    return timeit_


def CalcSqrtBeta(d, t, scale, R, ridge, delta, S_hat=1.0):
    return scale * (R * np.sqrt(d * np.log((1 + t / (ridge * d)) / delta)) + np.sqrt(ridge) * S_hat)


def argmax_reward(X, theta, invV, x_invVt_norm, do_not_ask=[], k=0):
    r"""
    Loop over all columns of X to solve this equation:

        \widehat{x} = \arg \min_{x \in X} x^T theta + k x^T V^{-1} x
    """
    # inv = np.linalg.inv
    # norm = np.linalg.norm
    # iV = np.linalg.inv(V)
    # rewards = [np.inner(X[:, c], theta) + k*np.inner(X[:, c], iV.dot(X[:, c]))
               # for c in range(X.shape[1])]
    # rewards = np.asarray(rewards)
    # return X[:, np.argmax(rewards)], np.argmax(rewards)
    sqrt = np.sqrt
    #utils.debug_print("OFUL28: do_not_ask = {}".format(do_not_ask))

    # MATLAB script: theta.T @ X + k*sqrt(beta)
    #rewards = X.T.dot(theta) + sqrt(k)*sqrt(beta)
    #rewards = X.dot(theta) + sqrt(k) * sqrt(beta)
    #utils.debug_print(X.shape)
    #utils.debug_print(theta.shape)
    rewards = np.dot(X, theta) + sqrt(k) * sqrt(x_invVt_norm)
    rewards[do_not_ask] = -np.inf
    return X[np.argmax(rewards),:], np.argmax(rewards)

@timeit(fn_name="get_feature_vectors")
def get_feature_vectors(butler):
    home_dir = '/Users/aniruddha'
    features = np.load('features_d100.npy'.format(home_dir))
    utils.debug_print("OFUL.py 120, features.shape = {}".format(features.shape))
    return features

@timeit(fn_name="get_hashing_functions")
def get_hashing_function():
    #try:
    #    with open('hashing_functions.pkl') as f:
    #        data = pickle.load(f)
    #except:
    #    raise ValueError('Current path:', os.getcwd())
    #from next.lib.hash import kjunutils, lsh_kjun_v3
    with open('hashing_functions.pkl') as f:
      index = pickle.load(f)

    index = hash.to_serializable(index)
    return index

class OFUL_Hashing:
    def initExp(self, butler, params=None, n=None, R=None, ridge=None,
                failure_probability=None):
        """
        initialize the experiment

        (int) n : number of arms
        (float) R : sub-Gaussian parameter, e.g. E[exp(t*X)]<=exp(t^2 R^2/2),
                    defaults to R=0.5 (satisfies X \in [0,1])
        (float) failure_probability : confidence
                imp note: delta
        (dict) params : algorithm-specific parameters (if none provided in
                        alg_list of init experiment, params=None)

        Expected output (comma separated):
          (boolean) didSucceed : did everything execute correctly
        """
        # setting the target matrix, a description of each target
        X = get_feature_vectors(butler)

        d = X.shape[1]  # number of dimensions in feature
        n = X.shape[0]

        #lsh = get_hashing_function()
        #butler.db.lsh = lsh
        #lsh['projections_all'] = []
        #lsh['projections'] = []
        #lsh = 'whatever'
        #lsh.projections_all = []
        #utils.debug_print('lsh: ', lsh)
        #utils.debug_print('lsh.keys(): ', lsh.keys())

        #lambda_ = ridge
        lambda_ = 1.0
        R = 1.0

        to_save = {#'X': X.tolist(),
                   'R': R, 'd': d, 'n': n, 'c1': 4.0, 'max_dist_comp': 500,
                   'lambda_': lambda_,
                   'total_pulls': 0.0,
                   'rewards': [],
                   'ask_indices': range(n),
                   'arms_pulled': [],
                   #'lsh': json.dumps(lsh),
                   'failure_probability': failure_probability}

        for name in to_save:
            butler.algorithms.set(key=name, value=to_save[name])

        # utils.debug_print('OFUL#L185')

        return True

    @timeit(fn_name='alg:getQuery')
    def getQuery(self, butler, participant_uid):
        """
        A request to ask which index/arm to pull

        Expected input:
          (list of int) do_not_ask_list : indices in {0,...,n-1} that the
                algorithm must not return. If there does not exist an index
                that is not in do_not_ask_list then any index is acceptable
                (this changes for each participant so they are not asked the
                same question twice)

        Expected output (comma separated):
          (int) target_index : idnex of arm to pull (in 0,n-1)

         particpant_doc is butler.participants corresponding to this
         participant

        if we want, we can find some way to have different arms
        pulled using the butler
        """

        initExp = butler.algorithms.get()
        X = get_feature_vectors(butler) # np.asarray(initExp['X'], dtype=float)

        # Scott: possible modification: if num_t
        participant_args = butler.participants.get(uid=participant_uid)
        # utils.debug_print('pargs: ', participant_args   )
        #if participant_args in [None, {}]:
        # utils.debug_print(participant_args)
        if participant_args is None:
            participant_args.update({'num_tries': 0, 'do_not_ask': []})
            #butler.participants.set(key='participant_{}'.format(participant_uid), value=participant_args)
            butler.participants.set_many(uid=participant_uid, key_value_dict=participant_args)
        # utils.debug_print('pargs.keys(): ', participant_args.keys())
        '''
        if 'invV_filename' not in participant_args.keys():
            ask_indices = range(X.shape[0])
            #ask_indices = [x for x in ask_indices if x !=participant_args['participant_args']['i_hat'] ]
            ask_indices = [x for x in ask_indices if x != participant_args['i_hat']]
            d = {'invV_filename': 'invV_{}.npy'.format(time.time() * 100),
                 #np.eye(initExp['d']) / initExp['lambda_'],
                 'beta': np.ones(X.shape[0]) / initExp['lambda_'],
                 't': 1,
                 'ask_indices': ask_indices,
                 'b': np.zeros(initExp['d']),
                 'participant_uid': participant_uid}
            participant_args.update(d)
            utils.debug_print('d =', initExp['d'])
            utils.debug_print('lambda =', initExp['lambda_'])
            invV = np.eye(initExp['d']) / initExp['lambda_']
            np.save(participant_args['invV_filename'], invV)

            butler.participants.set_many(uid=participant_uid, key_value_dict=participant_args)
        # if not 'theta_star' in participant_args:
        #     i_star = participant_args['i_star']
        #     d = {'reward': calc_reward(i_hat, X[:, i_star], R=reward_coeff
        #          * initExp['R']),
        #          'theta_star': (X[:, i_star])}
        #     participant_args.update(d)
        #     butler.participants.set_many(uid=participant_args['participant_uid'],
        #                             key_value_dict=participant_args)'''
        if 'invV' not in participant_args.keys():
            ask_indices = range(X.shape[0])
            # ask_indices = [x for x in ask_indices if x !=participant_args['participant_args']['i_hat'] ]
            ask_indices = [x for x in ask_indices if x != participant_args['i_hat']]
            invV = np.eye(initExp['d']) / initExp['lambda_']
            d = {'invV': invV,
                 # np.eye(initExp['d']) / initExp['lambda_'],
                 'x_invVt_norm': np.ones(X.shape[0]) / initExp['lambda_'],
                 't': 1,
                 'ask_indices': ask_indices,
                 'b': np.zeros(initExp['d']),
                 'participant_uid': participant_uid}
            participant_args.update(d)
            utils.debug_print('d =', initExp['d'])
            utils.debug_print('lambda =', initExp['lambda_'])
            #np.save(participant_args['invV_filename'], invV)

            butler.participants.set_many(uid=participant_uid, key_value_dict=participant_args)

        if 'theta_hat' not in participant_args.keys():
            # unsure if below needs to be i_hat or i_init (believe to be i_init)
            # i_hat is passed through choice of first image
            d = {'theta_hat': X[participant_args['i_hat'], :]}
            participant_args.update(d)
            butler.participants.set_many(uid=participant_args['participant_uid'],
                                         key_value_dict=participant_args)
            butler.participants.append(uid=participant_args['participant_uid'],
                                         key='do_not_ask', value=participant_args['i_hat'])

        # Figure out what query to ask
        t = participant_args['t']
        # scale = 1.0
        # scale = 1e-5
        scale = 0.0
        #c1 = participant_args['c1']
        utils.debug_print('initExp.keys(): ', initExp.keys())
        c1 = initExp['c1']
        #lsh = hash.from_serializable(initExp['lsh'])
        #lsh = hash.from_serializable(butler.db.lsh)
        lsh = butler.db.lsh
        max_dist_comp = initExp['max_dist_comp']
        index_array = range(X.shape[0])
        d = initExp['d']
        #log_div = (1 + t * 1.0/initExp['lambda_']) * 1.0 / initExp['failure_probability']
        #k = initExp['R'] * np.sqrt(initExp['d'] * np.log(log_div)) + np.sqrt(initExp['lambda_'])
        sqrt_beta = CalcSqrtBeta(d, t, scale, initExp['R'], initExp['lambda_'], initExp['failure_probability'])

        invV = np.array(participant_args['invV'])
        #invV = np.load(participant_args['invV_filename'])
        x_invVt_norm = np.array(participant_args['x_invVt_norm'])

        do_not_ask = butler.participants.get(uid=participant_args['participant_uid'],
                                             key='do_not_ask')

        validinds = np.setdiff1d(index_array, do_not_ask).astype('int')

        theta_hat = np.array(participant_args['theta_hat'])
        #arm_x, i_x = argmax_reward(X, theta_hat, invV, x_invVt_norm,
        #                           do_not_ask=do_not_ask, k=sqrt_beta)


        min_sqrt_eig = 1/np.sqrt(initExp['lambda_'])
        query = np.zeros((d + d ** 2, 1), 'float32')
        query[:d, 0] = theta_hat
        query[d:] = invV.reshape(d ** 2, 1) * (np.sqrt(sqrt_beta) / 4 / c1 / min_sqrt_eig)

        foundSet, foundListTuple = lsh.FindUpto(query, max_dist_comp, randomize=True,
                                                     invalidSet=do_not_ask)

        #utils.debug_print('foundListTuple: ', foundListTuple)
        #utils.debug_print('len(idx_ary): ', len(index_array))

        utils.debug_print(np.max([x[0] for x in foundListTuple]))

        foundList = [index_array[x[0]] for x in foundListTuple]
        foundList = np.intersect1d(foundList, validinds)

        sub_X = X[foundList, :]

        term1 = np.sum(sub_X * np.dot(sub_X, invV), axis=1)
        term2 = np.dot(sub_X, theta_hat)

        total = term2 + (np.sqrt(sqrt_beta) / 4 / c1 / min_sqrt_eig) * term1
        i_x = foundList[np.argmax(total)]

        butler.participants.append(uid=participant_args['participant_uid'],
                                   key='do_not_ask', value=i_x)

        # reward = calc_reward(arm_x, np.array(participant_args['theta_star']),
        #                      R=reward_coeff * initExp['R'])
        # # allow reward to propograte forward to other functions; it's
        # # used later
        # participant_args['reward'] = reward

        # for key in participant_args:
        #     butler.participants.set(uid=participant_args['participant_uid'],
        #                             key=key, value=participant_args[key])
        return i_x, participant_args

    @timeit(fn_name='alg:processAnswer')
    def processAnswer(self, butler, target_id=None,
                      target_reward=None, participant_doc=None):
        """
        reporting back the reward of pulling the arm suggested by getQuery

        Expected input:
          (int) target_index : index of arm pulled
          (int) target_reward : reward of arm pulled

        Expected output (comma separated):
          (boolean) didSucceed : did everything execute correctly
        """
        if target_id is None:
            return True

        args = butler.algorithms.get()
        # utils.debug_print('in OFUL, p_doc: ', participant_doc)
        butler.participants.increment(uid=participant_doc['participant_uid'], key='t')

        # this makes sure the reward propogates from getQuery to processAnswer
        reward = target_reward
        i_hat = participant_doc['i_hat']
        # theta_star = np.array(participant_args['theta_star'])
        X = get_feature_vectors(butler) # np.asarray(args['X'], dtype=float)
        b = np.array(participant_doc['b'], dtype=float)
        ask_indices = participant_doc['ask_indices']
        invV = np.array(participant_doc['invV'], dtype=float)
        #invV = np.load(participant_doc['invV_filename'])
        x_invVt_norm = np.array(participant_doc['x_invVt_norm'], dtype=float)

        #arm_pulled = X[:, target_id]
        arm_pulled = X[target_id, :]
        u = invV.dot(arm_pulled)
        invV -= np.outer(u, u) / (1 + np.inner(arm_pulled, u))

        #x_invVt_norm -= (X.T.dot(u))**2 / (1 + np.inner(arm_pulled, u))#x_invVt_norm[target_id])
        x_invVt_norm -= np.dot(X, u)**2/ (1 + np.inner(arm_pulled, u))#x_invVt_norm[target_id])

        b += reward * arm_pulled
        theta_hat = X[i_hat, :] + invV.dot(b)

        # save the results
        d = {#'invV': invV,
             'x_invVt_norm': x_invVt_norm,
             'b': b,
             'invV': invV,
             'theta_hat':theta_hat}
        participant_doc.update(d)

        #np.save(participant_doc['invV_filename'], invV)

        butler.participants.set_many(uid=participant_doc['participant_uid'],
                                     key_value_dict=participant_doc)
        return True

    def getModel(self, butler):
        """
        uses current model to return empirical estimates with uncertainties

        Expected output:
          (list float) mu : list of floats representing the emprirical means
          (list float) prec : list of floats representing the precision values
                              (or standard deviation)
        """
        # TODO: I can't see the results without this
        # (and we also need to change the label name if we want to see results,
        # correct?)
        return 0.5  # mu.tolist(), prec

