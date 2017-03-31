"""
OFUL X9 with Hashing
"""

from __future__ import division
import numpy as np
import next.utils as utils
from next.lib.bandits import banditclass as bc

class MyAlg:
    app_id = 'ImageSearch'
    def __init__(self):
        self.alg_id = 'OFUL'

    # InitExp only needs to initialize the dashboard if it hasn't already been done
    def initExp(self, butler, params=None, n=None, R=None, ridge=None,
                failure_probability=None):
        if butler.dashboard.get(key='plot_data') is None:
            butler.dashboard.set(key='plot_data', value=[])

        return True

    # getQuery does the following:
    #   - pull _bo_expected_rewards from butler
    #   - take the argmax of the rewards
    #   - update the _bo_do_no_ask list
    # Only getQuery should update do not ask as processAnswer is asynchronous and therefor not reliable
    def getQuery(self, butler, participant_uid, first_query_flag=False):
        utils.debug_print('Running OFUL')
        if first_query_flag:
            return 0
        else:
            expected_rewards = np.asarray(butler.participants.get(uid=participant_uid, key='_bo_expected_rewards'))
            do_not_ask = butler.participants.get(uid=participant_uid, key='_bo_do_not_ask')
            utils.debug_print('dna', do_not_ask)
            expected_rewards[np.asarray(do_not_ask)] = -np.inf
            i_x = np.argmax(expected_rewards)
            butler.participants.append(uid=participant_uid,
                                       key='_bo_do_not_ask', value=i_x)
            return i_x

    # Process answer has two parts:
    # Init: this is when the user clicks the initial target image. In this case, we pull precomputed order of arms and
    #       initialize estimated_rewards.
    # Update: this is when we get a Yes/No answer. Here the model parameters: b, invVt, theta_hat all must be updated
    def processAnswer(self, butler, target_id=None,
                      target_reward=None, participant_uid=None):
        if not target_id:
            target_id = butler.participants.get(uid=participant_uid, key='i_hat')

            n = 50025
            expected_rewards = np.ones(n)*-np.inf
            NN_order = np.load('NN_order.npy').tolist()
            expected_rewards[NN_order[target_id]] = range(0,50)[::-1]

            R = 1.
            ridge = 1.
            scale = 0.1
            delta = 0.1
            S_hat = 1.

            bandit_context = {
                '_bo_expected_rewards': expected_rewards,
                '_bo_do_not_ask': [target_id],
                'R': R,
                'ridge': ridge,
                'scale': scale,
                'delta': delta,
                'S_hat': S_hat
            }
            butler.participants.set_many(uid=participant_uid, key_value_dict=bandit_context)

            task_args = {
                'participant_uid': participant_uid,
                'target_id': target_id,
            }

            butler.job('modelInit', task_args, ignore_result=True)

            return True

        task_args = {
            'participant_uid': participant_uid,
            'target_id': target_id,
            'target_reward': target_reward
        }

        butler.job('modelUpdateHash', task_args, ignore_result=True, time_limit=300)

        return True

    def modelInit(self, butler, task_args):
        participant_uid = task_args['participant_uid']
        target_id = task_args['target_id']

        X = butler.db.X

        n = X.shape[0]
        d = X.shape[1]

        t = 1
        shifted = False


        scale = butler.participants.get(uid=participant_uid, key='scale')
        R = butler.participants.get(uid=participant_uid, key='R')
        ridge = butler.participants.get(uid=participant_uid, key='ridge')
        delta = butler.participants.get(uid=participant_uid, key='delta')
        S_hat = butler.participants.get(uid=participant_uid, key='S_hat')
        sqrt_beta = scale * (R * np.sqrt(d * np.log((1 + t / (ridge * d)) / delta)) + np.sqrt(ridge) * S_hat)

        invVt = np.eye(d) / ridge
        thetahat = X[target_id, :]
        b = np.zeros(len(thetahat))

        X_invVt_norm_sq = np.sum(X * X, axis=1) / ridge
        est_rewards = np.dot(X, thetahat) + sqrt_beta * np.sqrt(X_invVt_norm_sq)

        bandit_context = {
            'invVt': invVt,
            't': t,
            '_bo_expected_rewards': est_rewards,
            'shifted': shifted,
            'init_arm': target_id,
            'X_invVt_norm_sq': X_invVt_norm_sq,
            'd': d,
            'n': n,
            'b': b
        }


        butler.participants.set_many(uid=participant_uid, key_value_dict=bandit_context)
        # utils.debug_print('expected_rewards: ', est_rewards)
        utils.debug_print('... done initializing')

        return True

    def modelUpdateHash(self, butler, task_args):
        target_id = task_args['target_id']
        target_reward = task_args['target_reward']
        participant_uid = task_args['participant_uid']

        X = butler.db.X


        participant_doc = butler.participants.get(uid=participant_uid)
        # utils.debug_print('keys in pdoc: ', participant_doc.keys())

        # scale = butler.participants.get(uid=participant_uid, key='scale')
        # R = butler.participants.get(uid=participant_uid, key='R')
        # ridge = butler.participants.get(uid=participant_uid, key='ridge')
        # delta = butler.participants.get(uid=participant_uid, key='delta')
        # S_hat = butler.participants.get(uid=participant_uid, key='S_hat')

        # invVt = butler.participants.get(uid=participant_uid, key='invVt')
        # t = butler.participants.get(uid=participant_uid, key='t')
        # shifted = butler.participants.get(uid=participant_uid, key='shifted')
        # init_arm = butler.participants.get(uid=participant_uid, key='init_arm')
        # X_invVt_norm_sq = butler.participants.get(uid=participant_uid, key='X_invVt_norm_sq')
        # b = butler.participants.get(uid=participant_uid, key='b')
        # d = butler.participants.get(uid=participant_uid, key='d')

        scale = participant_doc['scale']
        R = participant_doc['R']
        ridge = participant_doc['ridge']
        delta = participant_doc['delta']
        S_hat = participant_doc['S_hat']

        invVt = participant_doc['invVt']
        t = participant_doc['t']
        shifted = participant_doc['shifted']
        init_arm = participant_doc['init_arm']
        X_invVt_norm_sq = participant_doc['X_invVt_norm_sq']
        b = participant_doc['b']
        d = participant_doc['d']

        xt = X[target_id, :]
        b += target_reward* xt
        tempval1 = np.dot(invVt, xt)
        tempval2 = np.dot(tempval1, xt)

        X_invVt_norm_sq = X_invVt_norm_sq - (np.dot(X, tempval1) ** 2) / (1 + tempval2)
        invVt = invVt - np.outer(tempval1, tempval1) / (1 + tempval2)
        if shifted:
            thetahat = np.dot(invVt, b) + X[init_arm,:]
        else:
            thetahat = np.dot(invVt, b)

        t += 1
        sqrt_beta = scale * (R * np.sqrt(d * np.log((1 + t / (ridge * d)) / delta)) + np.sqrt(ridge) * S_hat)


        est_rewards = np.dot(X, thetahat) + sqrt_beta * np.sqrt(X_invVt_norm_sq)

        bandit_context = {
            'invVt': invVt,
            't': t,
            '_bo_expected_rewards': est_rewards,
            'X_invVt_norm_sq': X_invVt_norm_sq,
            'b': b
        }

        butler.participants.set_many(uid=participant_uid, key_value_dict=bandit_context)


        if target_reward < 0:
            target_reward = 0

        update_plot_data = {'rewards': target_reward,
                            'participant_uid': participant_uid,
                            'initial_arm': init_arm,
                            'arm_pulled': target_id,
                            'alg': self.alg_id,
                            'time': t}

        butler.dashboard.append(key='plot_data', value=update_plot_data)

        return True

    def getModel(self, butler):
        """
        Return cumulative sum of current rewards
        """
        import pandas as pd
        plot_data = butler.dashboard.get(key='plot_data')
        print('plot_data: ', plot_data)
        if plot_data is not None:
            # data is a list of dicts with keys in bandit_context['plot_data']
            df = pd.DataFrame(plot_data)
            utils.debug_print('df: ', df)
            df = df.pivot_table(columns='initial arm', index='time', value='reward', aggfunc=np.mean)

            # return the right python builtin which is a dict with values of list
            d = dict(df)
            d = {key: list(value) for key, value in d.items()}
        else:
            d = {}
        return d
