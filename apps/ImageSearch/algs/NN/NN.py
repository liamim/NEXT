"""
Bandits
"""

from __future__ import division
import numpy as np
import next.utils as utils
import time

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


class NN:
    def __init__(self):
        self.alg_id = 'NN'

    def load_and_save_numpy(self, butler, filename, property_name, load_lib):
        if not butler.memory.exists(property_name):
            from StringIO import StringIO
            import StringIO

            utils.debug_print('loading file: %s'%(filename))
            data = np.load(filename)

            utils.debug_print('serialising %s'%property_name)
            s = StringIO.StringIO()
            np.save(s, data)
            utils.debug_print('storing %s'%property_name)
            butler.memory.set_file(property_name, s)
            data = ""
            s = ""

    def initExp(self, butler, params=None, n=None, R=None, ridge=None,
                failure_probability=None):
        load_lib = False
        self.load_and_save_numpy(butler, filename='features_d1000.npy', property_name='features', load_lib=load_lib)

        if butler.dashboard.get(key='plot_data') is None:
            butler.dashboard.set(key='plot_data', value=[])

        return True

    @timeit(fn_name='alg:getQuery')
    def getQuery(self, butler, participant_uid):
        utils.debug_print('Running NN')
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


            x0 = X[target_id,:]
            expected_rewards = X.dot(x0)
            utils.debug_print('ten biggest exp rews: ', np.sort(expected_rewards)[::-1][:10])
            utils.debug_print('ten biggest indices: ', np.argsort(expected_rewards)[::-1][:10])
            expected_rewards[target_id] = -np.inf
            bandit_context = {}
            bandit_context['init_arm'] = target_id
            bandit_context['_bo_expected_rewards'] = expected_rewards
            bandit_context['_bo_do_not_ask'] = [target_id]
            butler.participants.set_many(uid=participant_uid, key_value_dict=bandit_context)

            return True

        return True


    def getModel(self, butler):
        """
        Return cumulative sum of current rewards
        """
        import pandas as pd
        plot_data = butler.algorithms.get(key='plot_data')
        utils.debug_print('plot data in get modeL: ', plot_data)
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

@timeit(fn_name="get_feature_vectors")
def get_feature_vectors(butler):
    return np.load(butler.memory.get_file('features'))
    # home_dir = '/Users/aniruddha'
    # features = np.load('features_d100.npy'.format(home_dir))
    # # utils.debug_print("OFUL.py 120, features.shape = {}".format(features.shape))
    # return features
