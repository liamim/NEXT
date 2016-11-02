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

from next.lib.hash import lsh_kjun_v3
from next.lib.bandits import banditclass as bc


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
    # utils.debug_print('loading X ...')
    return np.load(butler.memory.get_file('features'))
    # home_dir = '/Users/aniruddha'
    # features = np.load('features_d100.npy'.format(home_dir))
    # # utils.debug_print("OFUL.py 120, features.shape = {}".format(features.shape))
    # return features

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

        return True

    @timeit(fn_name='alg:getQuery')
    def getQuery(self, butler, participant_uid):
        expected_rewards = np.asarray(butler.participants.get(uid=participant_uid, key='_bo_expected_rewards'))
        do_not_ask = butler.participants.get(uid=participant_uid, key='_bo_do_not_ask')
        # utils.debug_print('dna: ', do_not_ask)
        expected_rewards[np.asarray(do_not_ask)] = -np.inf
        i_x = np.argmax(expected_rewards)
        # utils.debug_print('add %d to dna'%(i_x))
        butler.participants.append(uid=participant_uid,
                                   key='_bo_do_not_ask', value=i_x)
        return i_x

    @timeit(fn_name='alg:processAnswer')
    def processAnswer(self, butler, target_id=None,
                      target_reward=None, participant_uid=None):
        if not target_id:
            participant_doc = butler.participants.get(uid=participant_uid)
            target_id = butler.participants.get(uid=participant_uid, key='i_hat')
            # utils.debug_print('pargs in processAnswer:', participant_doc)
            X = get_feature_vectors(butler)
            lsh = np.load(butler.memory.get_file('lsh')).tolist()
            lsh.projections_all = np.load(butler.memory.get_file('projections_all'))

            utils.debug_print('find upto function: ', lsh.FindUpto)
            opts = bc.bandit_init_options()
            opts['lsh'] = lsh
            opts['lsh_index_array'] = np.load(butler.memory.get_file('lsh_index_array'))
            utils.debug_print('lsh index array: ', opts['lsh_index_array'])
            bandit_context = bc.bandit_init('ofulx9_lsh', target_id, X, opts=opts)
            butler.participants.set_many(uid=participant_uid, key_value_dict=bandit_context)

            return True

        task_args = {
            'target_id': target_id,
            'target_reward': target_reward,
            'participant_uid': participant_uid
        }

        butler.job('modelUpdate', task_args, ignore_result=True)

        return True

    def modelUpdate(self, butler, task_args):
        target_id = task_args['target_id']
        target_reward = task_args['target_reward']
        participant_uid = task_args['participant_uid']

        X = get_feature_vectors(butler)
        lsh = np.load(butler.memory.get_file('lsh')).tolist()
        lsh.projections_all = np.load(butler.memory.get_file('projections_all'))

        participant_doc = butler.participants.get(uid=participant_uid)
        bandit_context = bc.bandit_extract_context(participant_doc)
        i_hat = butler.participants.get(uid=participant_uid, key='i_hat')
        bc.bandit_update(bandit_context, X, i_hat, target_reward, {'lsh': lsh})

        participant_doc.update(bandit_context)

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


