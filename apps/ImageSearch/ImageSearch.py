import numpy as np
import time
from decorator import decorator
from line_profiler import LineProfiler

@decorator
def profile_each_line(func, *args, **kwargs):
    profiler = LineProfiler()
    profiled_func = profiler(func)
    retval = None
    try:
        retval = profiled_func(*args, **kwargs)
    finally:
        profiler.print_stats()
    return retval


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


        t0 = time.time()
        if 'targetset' in args['targets'].keys():
            n = len(args['targets']['targetset'])
            new_targetset = args['targets']['targetset']
            self.TargetManager.set_targetset(butler.exp_uid, new_targetset)
        else:
            n = args['targets']['n']
            X = np.array(args['features']['matrix'])
            np.save('features.npy', X)

        args['n'] = n

        if 'labels' in args['rating_scale'].keys():
            labels = args['rating_scale']['labels']

        algorithm_keys = ['n', 'failure_probability', 'R']
        alg_data = {}
        for key in algorithm_keys:
            alg_data[key] = args[key]
        alg_data['ridge'] = args['ridge']

        init_algs(alg_data)

        num_algs_pulled = {alg['alg_id']: 0 for alg in args['alg_list']}
        possible_target_indices = ['2226', '35793', '36227']
        num_arms_pulled = {arm: 0 for arm in possible_target_indices}
        utils.debug_print('num_arms_pulled: ', num_arms_pulled)
        args['num_algs_pulled'] = num_algs_pulled
        for alg in args['alg_list']:
            args[alg['alg_id']] = {}
            args[alg['alg_id']]['num_starting_points_pulled'] = num_arms_pulled.copy()

        del args['targets']
        del args['feature_filenames']
        return args

    # @timeit(fn_name='myApp.py:getQuery')
    @profile_each_line
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
        utils.debug_print("hi mom")
        t0 = time.time()
        exp_uid = butler.exp_uid
        # t1 = time.time()
        participant_uid = args.get(u'participant_uid')  # , exp_uid + '_{}'.format(np.random.randint(1e6)))
        # t2 = time.time()
        # participant_doc = butler.participants.get(uid=participant_uid)
        num_tries = butler.participants.get(uid=participant_uid, key='num_tries')
        # t3 = time.time()

        # utils.debug_print('time to get exp_uid: ', t1 - t0)
        # utils.debug_print('time to get p_uid: ', t2 - t1)
        # utils.debug_print('time to get p_doc: ', t3 - t2)
        experiment_dict = butler.experiment.get(key='args')
        alg_id = butler.participants.get(uid=participant_uid, key='alg_id')
        utils.debug_print('starting_arms_pulls for ', alg_id, experiment_dict[alg_id]['num_starting_points_pulled'])

        if not num_tries or num_tries == 0:
            # utils.debug_print('came here 2')
            # utils.debug_print('num_tries was empty or it was 0, choosing start options')
            # t5 = time.time()
            # target_indices = random.sample(range(butler.experiment.get(key='args')['n']), 9)  # 9 here means "show 9 + 1 random queries at the start"
            # target_indices = [40767]
            # target_indices = [4050, 2959, 2226]
            # target_indices = [35828] # a super hard starting point
            # target_indices = [35793]
            butler.participants.set(uid=participant_uid, key='ntries', value=1)

            num_starting_points_pulled = experiment_dict[alg_id]['num_starting_points_pulled']
            next_arm = min(num_starting_points_pulled, key=num_starting_points_pulled.get)
            num_starting_points_pulled[next_arm] += 1
            experiment_dict[alg_id]['num_starting_points_pulled'] = num_starting_points_pulled
            butler.experiment.set(key='args', value=experiment_dict)

            # possible_target_indices = [2226, 35793, 36227]#, 1234]  # red boot, hard prewalker, asics and
            # target_indices = [np.random.choice(possible_target_indices)]
            target_indices = [int(next_arm)]
            target_instructions = {2226: 'Pick red boots', 35793: 'Pick only shoes for babies/toddlers',
                                   36227: 'Pick only ASICS branded shoes'}
                                   #1234: 'Pick dark colored short boots (ankle boots)'}
            # target_indices = [2226]  # red boot, hard prewalker, asics and
            # target_instructions = {2226: 'Pick red boots'}  #, 35793: 'Pick only shoes for small children', 36227: 'Pick only ASICS branded shoes', 1234: 'Pick dark colored short boots (ankle boots)'}
            targets_list = [{'index': i, 'target': self.TargetManager.get_target_item(exp_uid, i),
                             'instructions': target_instructions[i]} for i in
                            target_indices]
            # t6 = time.time()
            return_dict = {'initial_query': True, 'targets': targets_list,
                           # 'instructions': butler.experiment.get(key='args')['instructions']}
                           'instructions': 'In this experiment, we will show you a total of 50 images. For each image, you will be asked if it is similar to the image currently shown. To make your judgement, please look at the image and read the description below. Click on the image when you are ready to proceed. Allow for 15-20 seconds after clicking the initial image. '}  # 'instructions': 'Please select the image and allow for 15-20 seconds after selecting initial image: '
            # t7 = time.time()

            # utils.debug_print('time to get N: ', t5 - t3)
            # utils.debug_print('time to init target_list: ', t6 - t5)
            # utils.debug_print('time to set return dict: ', t7 - t6)
        else:
            # utils.debug_print('came here 3')
            # t8 = time.time()
            # i_x, participant_args = alg({'participant_uid': participant_uid})
            i_x = alg({'participant_uid': participant_uid})
            # utils.debug_print('keys() after initial query: ', participant_args.keys())
            # i_x = alg({'participant_uid': participant_uid})
            # t9 = time.time()
            # t10 = time.time()
            # butler.participants.set(key=participant_uid, value=participant_args)
            # t11 = time.time()
            target = self.TargetManager.get_target_item(exp_uid, i_x)
            # t12 = time.time()
            targets_list = [{'index': i_x, 'target': target}]
            # t13 = time.time()
            init_index = butler.participants.get(uid=participant_uid, key="i_hat")
            # t14 = time.time()
            init_target = self.TargetManager.get_target_item(exp_uid, init_index)
            # t15 = time.time()
            # t16 = time.time()
            # t = butler.participants.get(uid=participant_uid, key="num_tries")
            t = butler.participants.get(uid=participant_uid, key='ntries')
            utils.debug_print('pargs.num_tries: ', t)
            # counterString = '{t}/50'.format(t=t)
            counterString = 'Query image'
            return_dict = {'initial_query': False, 'targets': targets_list, 'main_target': init_target,
                           # 'instructions': butler.experiment.get(key='args')['instructions']} # changed query_instructions to instructions
                           'instructions': 'Is this the kind of image you are looking for?',
                           'count': counterString
                           }
            butler.participants.set(uid=participant_uid, key='ntries', value=t + 1)

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

    # @timeit(fn_name='myApp:processAnswer')
    @profile_each_line
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
        participant_uid = butler.queries.get(uid=args['query_uid'], key='participant_uid')
        butler.participants.increment(uid=participant_uid, key='num_tries')
        if args['initial_query']:
            # utils.debug_print('I should be starting here')
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

    # def getModel(self, exp_uid, alg_response, args_dict, butler):
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

    def chooseAlg(self, butler, args):
        experiment_dict = butler.experiment.get()
        # utils.debug_print('keys() of exp_dict: ', experiment_dict.keys())
        # utils.debug_print('exp_dict[args][num_algs_pulls]: ', experiment_dict['args']['num_algs_pulled'])
        num_algs_pulled = experiment_dict['args']['num_algs_pulled']
        next_alg = min(num_algs_pulled, key=num_algs_pulled.get)
        num_algs_pulled[next_alg] += 1
        experiment_dict['args']['num_algs_pulled'] = num_algs_pulled
        # utils.debug_print('exp_dict[args][num_algs_pulls] after update: ', experiment_dict['args']['num_algs_pulled'])
        butler.experiment.set(key='args', value=experiment_dict['args'])
        alg_list = experiment_dict['args']['alg_list']
        val = np.random.choice(alg_list)
        # utils.debug_print('args[alg_list]: ', experiment_dict['args']['alg_list'])
        # utils.debug_print('random choice = ', val )
        for val in experiment_dict['args']['alg_list']:
            if val['alg_label'] == next_alg:
                return val
        # return next_alg
