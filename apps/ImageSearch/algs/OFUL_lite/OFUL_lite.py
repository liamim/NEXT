"""
Bandits
"""

from __future__ import division
import numpy as np
import next.utils as utils
import time
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


class OFUL_lite:
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
        # X = np.asarray(params['X'])
        # X = get_feature_vectors(butler)
        # X = butler.db.X
        #X = butler.db.get_features(butler.app_id, butler.exp_uid)
        #utils.debug_print('loading X')
        #X = np.load(butler.memory.get_file('features'))
        # theta_star = np.asarray(params['theta_star'])
        # d = X.shape[1]  # number of dimensions in feature
        # n = X.shape[0]

        #lambda_ = ridge
        # lambda_ = 1.0
        # R = 1.0

        # initial sampling arm
        # theta_hat = X[:, np.random.randint(X.shape[1])]
        # theta_hat = np.random.randn(d)
        # theta_hat /= np.linalg.norm(theta_hat)

        # to_save = {'R': R, 'd': d, 'n': n,
        #            'lambda_': lambda_,
        #            'total_pulls': 0.0,
        #            'rewards': [],
        #            'arms_pulled': [],
        #            'failure_probability': failure_probability}
        #
        # for name in to_save:
        #     butler.algorithms.set(key=name, value=to_save[name])

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

            opts = bc.bandit_init_options()
            bandit_context = bc.bandit_init('oful_light', target_id, X, opts=opts)
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

        participant_doc = butler.participants.get(uid=participant_uid)
        bandit_context = bc.bandit_extract_context(participant_doc)
        i_hat = butler.participants.get(uid=participant_uid, key='i_hat')
        bc.bandit_update(bandit_context, X, i_hat, target_reward, {'lsh': None})

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

@timeit(fn_name="get_feature_vectors")
def get_feature_vectors(butler):
    utils.debug_print('loading X (in oful_lite)')
    return np.load(butler.memory.get_file('features'))
    # home_dir = '/Users/aniruddha'
    # features = np.load('features_d100.npy'.format(home_dir))
    # # utils.debug_print("OFUL.py 120, features.shape = {}".format(features.shape))
    # return features
