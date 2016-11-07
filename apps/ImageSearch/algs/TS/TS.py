"""
Lazy Bandits
"""

from __future__ import division
import numpy as np
import next.utils as utils
import time
from StringIO import StringIO
import StringIO
from next.lib.bandits import banditclass as bc
import pandas as pd
import json

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


class TS:
    def __init__(self):
        self.alg_id = 'TS'

    def load_and_save_numpy(self, butler, filename, property_name, load_lib):
        if not butler.memory.exists(property_name):
            if load_lib:
                from next.lib.hash import kjunutils
                from next.lib.hash import lsh_kjun_v3
                from next.lib.hash import lsh_kjun_nonquad

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

        self.load_and_save_numpy(butler, filename='lsh_index_array.npy', property_name='lsh_index_array', load_lib=load_lib)

        self.load_and_save_numpy(butler, filename='projections_nonquad.npy', property_name='projections_nonquad', load_lib=load_lib)

        load_lib = False
        self.load_and_save_numpy(butler, filename='hash_object_nonquad.npy', property_name='lsh_non_quad', load_lib=load_lib)

        butler.algorithms.set(key='plot_data', value=[])

        if butler.dashboard.set(key='plot_data', value=[]) is None:
            butler.dashboard.set(key='plot_data', value=[])

        return True

    @timeit(fn_name='alg:getQuery')
    def getQuery(self, butler, participant_uid):
        utils.debug_print('Running TS')
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
            utils.debug_print('came in to init TS')
            participant_doc = butler.participants.get(uid=participant_uid)
            target_id = butler.participants.get(uid=participant_uid, key='i_hat')

            X = get_feature_vectors(butler)
            lsh = np.load(butler.memory.get_file('lsh_non_quad')).tolist()
            lsh.projections_all = np.load(butler.memory.get_file('projections_nonquad'))
            opts = bc.bandit_init_options()
            opts['lsh'] = lsh
            opts['lsh_index_array'] = np.load(butler.memory.get_file('lsh_index_array'))
            bandit_context = bc.bandit_init('ts_lsh', target_id, X, opts=opts)
            bandit_context['t'] = 0
            bandit_context['init_arm'] = target_id
            butler.participants.set_many(uid=participant_uid, key_value_dict=bandit_context)
            return True

        utils.debug_print('came in to run TS')
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
        lsh = np.load(butler.memory.get_file('lsh_non_quad')).tolist()
        lsh.projections_all = np.load(butler.memory.get_file('projections_nonquad'))

        participant_doc = butler.participants.get(uid=participant_uid)
        bandit_context = bc.bandit_extract_context(participant_doc)
        i_hat = butler.participants.get(uid=participant_uid, key='i_hat')
        bc.bandit_update(bandit_context, X, i_hat, target_reward, {'lsh': lsh})
        if target_reward < 0:
            target_reward = 0
        t = participant_doc[u't']
        update_plot_data = {'rewards': target_reward,
                            'participant_uid': participant_uid,
                            'initial_arm': participant_doc['init_arm'],
                            'arm_pulled': target_id,
                            'alg': self.alg_id,
                            'time': t}

        # butler.algorithms.append(key='plot_data', value=update_plot_data)
        butler.dashboard.append(key='plot_data', value=update_plot_data)
        # utils.debug_print('butler info: ', butler.exp_uid, butler.app_id, butler.alg_id, butler.alg_label)
        # butler.job('logData',
        #            json.dumps({'exp_uid': butler.exp_uid, 'args': {'plot_data': update_plot_data, 'logging': True}}))
        # butler.log('plot_data', json.dumps(update_plot_data))
        bandit_context['t'] = t + 1
        participant_doc.update(bandit_context)
        butler.participants.set_many(uid=participant_doc['participant_uid'],
                                     key_value_dict=participant_doc)

        # plot_data = butler.algorithms.get(key='plot_data')
        # plot_data = butler.dashboard.get(key='plot_data')
        # utils.debug_print('butler.algs.plot_data in modelUpdate: ', plot_data)
        return True

    def logData(self, exp_uid, args):
        utils.debug_print('came into logData')
        return True

    def getModel(self, butler):
        """
        Return cumulative sum of current rewards
        """
        # plot_data = butler.algorithms.get(key='plot_data')
        plot_data = butler.dashboard.get(key='plot_data')
        utils.debug_print('butler.algorithms.get(): ', butler.algorithms.get())
        # utils.debug_print('plot data in get model: ', plot_data)
        utils.debug_print('butler.algs.plot_data in getModel: ', plot_data)

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

        # num_starting_points = len(plot_data[''])
        # rewards = np.array(butler.participants.get(uid=participant_uid, key='rewards'))
        # rewards[rewards == -1] = 0
        # return np.cumsum(rewards)

@timeit(fn_name="get_feature_vectors")
def get_feature_vectors(butler):
    # utils.debug_print('loading X (lite)')
    return np.load(butler.memory.get_file('features'))
    # home_dir = '/Users/aniruddha'
    # features = np.load('features_d100.npy'.format(home_dir))
    # # utils.debug_print("OFUL.py 120, features.shape = {}".format(features.shape))
    # return features
