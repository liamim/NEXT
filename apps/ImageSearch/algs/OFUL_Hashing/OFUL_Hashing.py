"""
OFUL X9 with Hashing
"""

from __future__ import division
import numpy as np
import next.utils as utils
from next.lib.bandits import banditclass as bc

class OFUL_Hashing:
    def __init__(self):
        self.alg_id = 'OFUL_X9'

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
    def getQuery(self, butler, participant_uid):
        utils.debug_print('Running OFUL X9 Hashing')
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
            bandit_context = {'_bo_expected_rewards': expected_rewards, '_bo_do_not_ask': [target_id]}
            butler.participants.set_many(uid=participant_uid, key_value_dict=bandit_context)

            task_args = {
                'participant_uid': participant_uid,
                'target_id': target_id
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

        lsh = butler.db.lsh
        lsh.projections_all = butler.db.projections_all

        opts = bc.bandit_init_options()
        opts['lsh'] = lsh
        opts['lsh_index_array'] = butler.db.lsh_index_array
        opts['param2'] = 10.0 ** -4
        opts['max_dist_comp'] = 500

        bandit_context = bc.bandit_init('ofulx9_lsh', target_id, X, opts=opts)
        bandit_context['t'] = 0
        bandit_context['init_arm'] = target_id
        del bandit_context['_bo_do_not_ask']
        butler.participants.set_many(uid=participant_uid, key_value_dict=bandit_context)

        return True

    def modelUpdateHash(self, butler, task_args):
        target_id = task_args['target_id']
        target_reward = task_args['target_reward']
        participant_uid = task_args['participant_uid']

        X = butler.db.X
        lsh = butler.db.lsh
        lsh.projections_all = butler.db.projections_all
        participant_doc = butler.participants.get(uid=participant_uid)

        bandit_context = bc.bandit_extract_context(participant_doc)
        i_hat = butler.participants.get(uid=participant_uid, key='i_hat')
        bc.bandit_update(bandit_context, X, i_hat, target_reward, {'lsh': lsh})
        if target_reward < 0:
            target_reward = 0
        participant_doc.update(bandit_context)
        t = participant_doc['t']

        update_plot_data = {'rewards': target_reward,
                            'participant_uid': participant_uid,
                            'initial_arm': participant_doc['init_arm'],
                            'arm_pulled': target_id,
                            'alg': self.alg_id,
                            'time': t}

        butler.dashboard.append(key='plot_data', value=update_plot_data)

        bandit_context['t'] = t + 1
        participant_doc.update(bandit_context)
        del participant_doc['_bo_do_not_ask']
        butler.participants.set_many(uid=participant_doc['participant_uid'],
                                     key_value_dict=participant_doc)

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