import json
import numpy
import random
import numpy as np
import urllib2
import requests
import os
import time
import datetime
import pickle
# today = str(datetime.today())[:10]
today = '2016-11-04'

import next.apps.SimpleTargetManager
import next.utils as utils

# utils.debug_print('TTTqwe1', time.time())
# from next.lib.hash import lsh_kjun_v3
# utils.debug_print('TTTqwe2', time.time())
# from next.lib.hash import lsh_kjun_nonquad
# utils.debug_print('TTTqwe3', time.time())
# from next.lib.hash.kjunutils import *
# utils.debug_print('TTTqwe4', time.time())

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


class ImageSearch(object):
    def __init__(self, db):
        self.app_id = 'ImageSearch'
        self.TargetManager = next.apps.SimpleTargetManager.SimpleTargetManager(db)

    # def load_and_save_numpy(self, butler, filename, property_name):
    #     utils.debug_print('loading file: %s'%(filename))
    #     data = numpy.load(filename)
    #
    #     utils.debug_print('serialising %s'%property_name)
    #     s = StringIO.StringIO()
    #     np.save(s, data)
    #     utils.debug_print('storing %s'%property_name)
    #     butler.memory.set_file(property_name, s)
    #     data = ""
    #     s = ""

    def initExp(self, butler, init_algs, args):
        """
        This function is meant to store any additional components in the
        databse.

        Inputs
        ------
        exp_uid : The unique identifier to represent an experiment.
        exp_data : The keys specified in the app specific YAML file in the
                   initExp section.
        butler : The wrapper for database writes. See next/apps/Butler.py for
                 more documentation.

        Returns
        -------
        exp_data: The experiment data, potentially modified.
        """
        
        # utils.debug_print('loading features')
        # Lfeatures = numpy.load('features_d1000.npy')
        # self.load_and_save_numpy(butler, filename='features_d1000.npy', property_name='features')
        #
        # utils.debug_print('loading projections_all in',os.getpid())
        # Lprojections_all = numpy.load('projections_all.npy')
        # self.load_and_save_numpy(butler, filename='projections_all.npy', property_name='projections_all')
        
        # utils.debug_print('loading projs in',os.getpid())
        # Lprojs = numpy.load('projs.npy')
        
        # utils.debug_print('loading lsh in',os.getpid())
        # Llsh = numpy.load('hash_object.npy')
        # self.load_and_save_numpy(butler, filename='hash_object.npy', property_name='lsh')
        #
        # self.load_and_save_numpy(butler, filename='lsh_index_array.npy', property_name='lsh_index_array')
        #
        # self.load_and_save_numpy(butler, filename='projections_nonquad.npy', property_name='projections_nonquad')
        #
        # self.load_and_save_numpy(butler, filename='hash_object_nonquad.npy', property_name='lsh_non_quad')

        
        # utils.debug_print('serialising features')
        # s = StringIO.StringIO()
        # np.save(s,Lfeatures)
        # utils.debug_print('storing features')
        # butler.memory.set_file('features',s)
        # s=""
        
        # utils.debug_print('serialising projections_all')
        # s = StringIO.StringIO()
        # np.save(s,Lprojections_all)
        # Lprojections_all = None
        # utils.debug_print('storing all')
        # butler.memory.set_file('projections_all',s)
        
        # utils.debug_print('serialising projs')
        # s = StringIO.StringIO()
        # np.save(s,Lprojs)
        # Lprojs = None
        # utils.debug_print('storing projs')
        # butler.memory.set_file('projs',s)

        # utils.debug_print('serialising lsh')
        # s = StringIO.StringIO()
        # np.save(s,Llsh)
        # Llsh = None
        # utils.debug_print('storing projs')
        # butler.memory.set_file('lsh',s)
        
        # s=None
        # utils.debug_print('serialising lsh')
        # s = json.dumps(Llsh.tolist())
        # utils.debug_print('storing lsh')
        # butler.memory.set('lsh',s)

        
        
        t0 = time.time()
        if 'targetset' in args['targets'].keys():
            n = len(args['targets']['targetset'])
            new_targetset = args['targets']['targetset']
            self.TargetManager.set_targetset(butler.exp_uid, new_targetset)
        else:
            n = args['targets']['n']
            X = np.array(args['features']['matrix'])
            np.save('features.npy', X)
        t1 = time.time()
        args['n'] = n

        if 'labels' in args['rating_scale'].keys():
            labels = args['rating_scale']['labels']

        algorithm_keys = ['n', 'failure_probability', 'R']
        alg_data = {}
        for key in algorithm_keys:
            alg_data[key] = args[key]
        alg_data['ridge'] = args['ridge']
        t2 = time.time()
        init_algs(alg_data)
        t3 = time.time()
        # if False:
            # utils.debug_print('Timing in initExp: ')
            # utils.debug_print('time to get through if else: ', t1 - t0)
            # utils.debug_print('time to setup variables: ', t2 - t1)
            # utils.debug_print('time to initialize algorithm: ', t3 - t2)
        del args['targets']
        return args

    @timeit(fn_name='myApp.py:getQuery')
    def getQuery(self, butler, alg, args):
        """
        The function that gets the next query, given a query reguest and
        algorithm response.

        Inputs
        ------
        exp_uid : The unique identiefief for the exp.
        query_request :
        alg_response : The response from the algorithm. The algorithm should
                       return only one value, be it a list or a dictionary.
        butler : The wrapper for database writes. See next/apps/Butler.py for
                 more documentation.

        Returns
        -------
        A dictionary with a key ``target_indices``.

        TODO: Document this further
        """
        t0 = time.time()
        exp_uid = butler.exp_uid
        # t1 = time.time()
        participant_uid = args.get(u'participant_uid') #, exp_uid + '_{}'.format(np.random.randint(1e6)))
        # t2 = time.time()
        #participant_doc = butler.participants.get(uid=participant_uid)
        num_tries = butler.participants.get(uid=participant_uid, key='num_tries')
        # t3 = time.time()

        # utils.debug_print('time to get exp_uid: ', t1 - t0)
        # utils.debug_print('time to get p_uid: ', t2 - t1)
        # utils.debug_print('time to get p_doc: ', t3 - t2)

        if not num_tries or num_tries == 0:
            # utils.debug_print('came here 2')
            # utils.debug_print('num_tries was empty or it was 0, choosing start options')
            # t5 = time.time()
            #target_indices = random.sample(range(butler.experiment.get(key='args')['n']), 9)  # 9 here means "show 9 + 1 random queries at the start"
            #target_indices = [40767]
            #target_indices = [4050, 2959, 2226]
            # target_indices = [35828] # a super hard starting point
            # target_indices = [35793]
            butler.participants.set(uid=participant_uid, key='ntries', value=1)
            target_indices = [2226, 35793, 36227, 1234] # red boot, hard prewalker, asics and
            target_instructions = {2226: 'Pick red boots', 35793: 'Pick only shoes for small children', 36227: 'Pick only ASICS branded shoes', 1234: 'Pick dark colored short boots (ankle boots)'}
            # target_indices = [2226]  # red boot, hard prewalker, asics and
            # target_instructions = {2226: 'Pick red boots'}  #, 35793: 'Pick only shoes for small children', 36227: 'Pick only ASICS branded shoes', 1234: 'Pick dark colored short boots (ankle boots)'}
            targets_list = [{'index': i, 'target': self.TargetManager.get_target_item(exp_uid, i), 'instructions': target_instructions[i]} for i in
                            target_indices]
            # t6 = time.time()
            return_dict = {'initial_query': True, 'targets': targets_list,
                           #'instructions': butler.experiment.get(key='args')['instructions']}
                           'instructions': 'Please select an image and allow for 15-20 seconds after selecting initial image: '}
            # t7 = time.time()

            # utils.debug_print('time to get N: ', t5 - t3)
            # utils.debug_print('time to init target_list: ', t6 - t5)
            # utils.debug_print('time to set return dict: ', t7 - t6)
        else:
            # utils.debug_print('came here 3')
            # t8 = time.time()
            #i_x, participant_args = alg({'participant_uid': participant_uid})
            i_x = alg({'participant_uid': participant_uid})
            #utils.debug_print('keys() after initial query: ', participant_args.keys())
            #i_x = alg({'participant_uid': participant_uid})
            # t9 = time.time()
            # t10 = time.time()
            #butler.participants.set(key=participant_uid, value=participant_args)
            # t11 = time.time()
            target = self.TargetManager.get_target_item(exp_uid, i_x)
            # t12 = time.time()
            targets_list = [{'index': i_x, 'target': target}]
            # t13 = time.time()
            init_index = butler.participants.get(uid=participant_uid, key="i_hat")
            # t14 = time.time()
            init_target = self.TargetManager.get_target_item(exp_uid, init_index)
            # t15 = time.time()
            experiment_dict = butler.experiment.get(key='args')
            # t16 = time.time()
            # t = butler.participants.get(uid=participant_uid, key="num_tries")
            t = butler.participants.get(uid=participant_uid, key='ntries')
            utils.debug_print('pargs.num_tries: ', t)
            counterString = '{t}/50'.format(t=t)

            return_dict = {'initial_query': False, 'targets': targets_list, 'main_target': init_target,
                           #'instructions': butler.experiment.get(key='args')['instructions']} # changed query_instructions to instructions
                           'instructions': 'Is this the kind of image you are looking for?',
                           'count': counterString
                           }
            butler.participants.set(uid=participant_uid, key='ntries', value=t+1)

            # t17 = time.time()

            if 'labels' in experiment_dict['rating_scale']:
                labels = experiment_dict['rating_scale']['labels']
                return_dict.update({'labels': labels})

                # t18 = time.time()
                # utils.debug_print('time to update return dict with labels: ', t18 - t17)

                if 'context' in experiment_dict and 'context_type' in experiment_dict:
                    return_dict.update({'context': experiment_dict['context'],
                                        'context_type': experiment_dict['context_type']})
                    # t19 = time.time()
                    # utils.debug_print('time to update return dict with context and context type: ', t19 - t18)

            # utils.debug_print('time to run alg(): ', t9 - t8)
            # utils.debug_print('time to set p_args: ', t11 - t10)
            # utils.debug_print('time to get target: ', t12 - t11)
            # utils.debug_print('time to get target_list: ', t13 - t12)
            # utils.debug_print('time to get init_index: ', t14 - t13)
            # utils.debug_print('time to get init_target: ', t15 - t14)
            # utils.debug_print('time to get experiment_dict: ', t16 - t15)
        t1 = time.time()
        # utils.debug_print('app getQuery took: %f seconds'%(t1-t0))
        # bargs = butler.experiment.get(key='args')
        # utils.debug_print('bargs[alglist]: ', bargs['alg_list'])
        # utils.debug_print('len(bargs_alist) = ', len(bargs['alg_list']))
        # num_algs = len(bargs['alg_list'])
        # test_alg_label = []
        # for i in range(num_algs):
        #     test_alg_label += [bargs['alg_list'][i]['alg_label']]
        # utils.debug_print('test_alg_labels = ', test_alg_label)
        # utils.debug_print('Butler experiment bargs: ', bargs.keys())
        return return_dict

    @timeit(fn_name='myApp:processAnswer')
    def processAnswer(self, butler, alg, args):
        """
        Parameters
        ----------
        butler :
        alg:
        args:

        Returns
        -------
        dictionary with keys:
            alg_args: Keywords that are passed to the algorithm.
            query_update :

        For example, this function might return ``{'a':1, 'b':2}``. The
        algorithm would then be called with
        ``alg.processAnswer(butler, a=1, b=2)``
        """
        t1 = time.time()
        participant_uid = butler.queries.get(uid=args['query_uid'],key='participant_uid')
        butler.participants.increment(uid=participant_uid, key='num_tries')
        if args['initial_query']:
            #utils.debug_print('I should be starting here')
            initial_arm = args['answer']['initial_arm']
            butler.participants.set(uid=participant_uid, key="i_hat", value=initial_arm)
            alg({'participant_uid': participant_uid})
            t2 = time.time()
            utils.debug_print('app processAnswer took: %f seconds' % (t2 - t1))
            return {}
        query_uid = args['query_uid']
        target_id = butler.queries.get(uid=query_uid)['targets'][0]['index']
        target_reward = args['answer']['target_reward']

        alg({'target_id': target_id, 'target_reward': target_reward, 'participant_uid': participant_uid})
        # return query_update, alg_args_dict
        t2 = time.time()
        utils.debug_print('app processAnswer took: %f seconds' % (t2 - t1))
        # query = butler.queries.get(uid=args['query_uid'])
        # participant_uid = args.get(u'participant_uid')
        # butler.job('getModel',
        #            json.dumps({'participant_uid': participant_uid,
        #                        'args': {'alg_label': query['alg_label'], 'logging': True}
        #                        })
        #            )

        return {'target_id': target_id, 'target_reward': target_reward}

    #def getModel(self, exp_uid, alg_response, args_dict, butler):
    def getModel(self, butler, alg, args):
        return alg()

    def getStats(self, exp_uid, stats_request, dashboard, butler):
        """
        Get statistics to display on the dashboard.
        """
        utils.debug_print("came into getStats")
        stat_id = stats_request['args']['stat_id']
        task = stats_request['args']['params'].get('task', None)
        alg_label = stats_request['args']['params'].get('alg_label', None)
        functions = {'api_activity_histogram': dashboard.api_activity_histogram,
                     'compute_duration_multiline_plot': dashboard.compute_duration_multiline_plot,
                     'compute_duration_detailed_stacked_area_plot': dashboard.compute_duration_detailed_stacked_area_plot,
                     'response_time_histogram': dashboard.response_time_histogram,
                     'network_delay_histogram': dashboard.network_delay_histogram,
                     'cumulative_reward_plot': dashboard.cumulative_reward_plot}

        default = [self.app_id, exp_uid]
        args = {'api_activity_histogram': default + [task],
                'compute_duration_multiline_plot': default + [task],
                'compute_duration_detailed_stacked_area_plot': default + [task, alg_label],
                'response_time_histogram': default + [alg_label],
                'network_delay_histogram': default + [alg_label],
                'cumulative_reward_plot': default + [alg_label]}
        return functions[stat_id](*args[stat_id])

