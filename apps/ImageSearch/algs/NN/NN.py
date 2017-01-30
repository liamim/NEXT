"""
Nearest Neighbors
"""

from __future__ import division
import numpy as np
import next.utils as utils


class NN:
    def __init__(self):
        self.alg_id = 'NN'

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
        utils.debug_print('Running NN')
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
            n = 50025
            target_id = butler.participants.get(uid=participant_uid, key='i_hat')

            expected_rewards = np.ones(n) * -np.inf
            NN_order = np.load('NN_order.npy').tolist()
            expected_rewards[NN_order[target_id]] = range(0, 50)[::-1]
            store_dictionary = {
                                '_bo_do_not_ask': [target_id],
                                '_bo_expected_rewards': expected_rewards
                               }
            butler.participants.set_many(uid=participant_uid, key_value_dict=store_dictionary)

            return True
        return True


    def getModel(self, butler):
        """
        Return cumulative sum of current rewards
        """
        import pandas as pd
        plot_data = butler.dashboard.get(key='plot_data')
        # data is a list of dicts with keys in bandit_context['plot_data']
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

