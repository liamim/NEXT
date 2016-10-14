import json
import numpy
import random
import numpy as np
import urllib2
import requests
from StringIO import StringIO
import os
import time

import next.apps.SimpleTargetManager
import next.utils as utils
from scipy.io import loadmat


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

    #def initExp(self, exp_uid, exp_data, butler):
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
        utils.debug_print('ImageSearch.py#L54')
        if 'targetset' in args['targets'].keys():
            n = len(args['targets']['targetset'])

            #targetset = exp_data['args']['targets']['targetset']
            targetset = args['targets']['targetset']
            #feature_filenames = exp_data['args']['feature_filenames']
            feature_filenames = args['feature_filenames']

            target_filenames = [target['primary_description'] for target in targetset]
            #utils.debug_print("feature filenames = ", feature_filenames)
            #utils.debug_print("target filenames = ", target_filenames)
           
            #new_target_idx = [feature_filenames.index(target) for target in target_filenames]
            #new_targetset = []

            utils.debug_print("image search:48, beginning to download features")
            if False:
                #response = requests.get(exp_data['args']['features'])
                #utils.debug_print(args['features'])
                response = requests.get(args['features'])
                #utils.debug_print(response.text)
                variables = np.load(StringIO(response.text))
                utils.debug_print("done downloading features")
                #X = variables['features_all'].T
                #np.save('features_NEXT.npy', X)
            else:
                home_dir = '/Users/aniruddha'
                string = 'wget -O features_d100.npy {1}'.format(home_dir, args['features'])
                r = os.system(string)
                ls = os.listdir('.')
                #utils.debug_print(ls)
                utils.debug_print('preparing URL to print')
                #utils.debug_print(r, string)
            #utils.debug_print("X.shape = {}, meaning {} shoes with {} features".format(X.shape, X.shape[1], X.shape[0]))

            
            #for col, target in zip(new_target_idx,
            #                       args['targets']['targetset']):
            #    # target['feature_vector'] = X[:, col].tolist()
            #    new_targetset += [target]

            #self.TargetManager.set_targetset(exp_uid, new_targetset)
            new_targetset = args['targets']['targetset']
            self.TargetManager.set_targetset(butler.exp_uid, new_targetset)

            # old code, expanded by the for-loop above
            # self.TargetManager.set_targetset(exp_uid,
            # [exp_data['args']['targets']['targetset'][i]
            # for i in new_target_idx])
        # Hasn't been tested yet
        else:
            #n = exp_data['args']['targets']['n']
            n = args['targets']['n']
            utils.debug_print("image search, 82")
            #X = np.array(exp_data['args']['features']['matrix'])
            X = np.array(args['features']['matrix'])
            np.save('features.npy', X)

        #exp_data['args']['n'] = n
        args['n'] = n
        #del exp_data['args']['features']
        #del exp_data['args']['targets']

        #if 'labels' in exp_data['args']['rating_scale'].keys():
        if 'labels' in args['rating_scale'].keys():
            #labels = exp_data['args']['rating_scale']['labels']
            labels = args['rating_scale']['labels']
            max_label = max(label['reward'] for label in labels)
            min_label = min(label['reward'] for label in labels)

        #R = exp_data['args']['rating_scale']['R']

        R = args['R']
        ridge = args['ridge']
        alg_data = {'R': R, 'ridge': ridge}
        algorithm_keys = ['n', 'failure_probability']
        for key in algorithm_keys:
            alg_data[key] = args[key]

        t1 = time.time()
        init_algs(alg_data)
        t2 = time.time()
        utils.debug_print('time to run getQuery: ', t2 - t1)
        #return exp_data, alg_data
        return args

    @timeit(fn_name='myApp.py:getQuery')
    #def getQuery(self, exp_uid, experiment_dict, query_request, alg_response, butler):
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

        exp_uid = butler.exp_uid
        participant_uid = args.get(u'participant_uid')#, exp_uid + '_{}'.format(np.random.randint(1e6)))
        #utils.debug_print(participant_uid)
        # participant_doc = butler.participants.get(uid=query_request['args']['participant_uid'])
        participant_doc = butler.participants.get(uid=participant_uid)
        # utils.debug_print("participant_doc in get query=", participant_doc)
        #print("participant_doc in get query=", participant_doc)
        if type(participant_doc) != dict:
            participant_doc = {}

        if 'num_tries' not in participant_doc.keys() or participant_doc['num_tries'] == 0:
            N = butler.experiment.get(key='args')['n']
            #target_indices = random.sample(range(N), 9)  # 9 here means "show 9 + 1 random queries at the start"
            target_indices = [4050, 2959, 2226]
            targets_list = [{'index': i, 'target': self.TargetManager.get_target_item(exp_uid, i)} for i in
                            target_indices]
            return_dict = {'initial_query': True, 'targets': targets_list,
                           'instructions': butler.experiment.get(key='args')['instructions']}
        else:
            args = {'participant_uid': participant_uid}
            #args.update(butler.participants.get(uid=participant_uid))
            t1 = time.time()
            i_x, participant_args = alg({'participant_uid': participant_uid})
            t2 = time.time()
            utils.debug_print('time to set keys: ', t2 - t1)
            butler.participants.set(key=participant_uid, value=participant_args)
            #target = self.TargetManager.get_target_item(exp_uid, alg_response)
            target = self.TargetManager.get_target_item(exp_uid, i_x)
            targets_list = [{'index': i_x, 'target': target}]

            #init_index = butler.participants.get(uid=query_request['args']['participant_uid'], key="i_hat")
            init_index = butler.participants.get(uid=participant_uid, key="i_hat")
            #init_target = self.TargetManager.get_target_item(exp_uid, init_index)
            init_target = self.TargetManager.get_target_item(exp_uid, init_index)

            experiment_dict = butler.experiment.get()

            return_dict = {'initial_query': False, 'targets': targets_list, 'main_target': init_target,
                           'instructions': butler.experiment.get(key='args')['instructions']} # changed query_instructions to instructions

            if 'labels' in experiment_dict['args']['rating_scale']:
                labels = experiment_dict['args']['rating_scale']['labels']
                return_dict.update({'labels': labels})

                if 'context' in experiment_dict['args'] and 'context_type' in experiment_dict['args']:
                    return_dict.update({'context': experiment_dict['args']['context'],
                                        'context_type': experiment_dict['args']['context_type']})
        return return_dict

    @timeit(fn_name='myApp:processAnswer')
    #def processAnswer(self, exp_uid, query, answer, butler):
    def processAnswer(self, butler, alg, args):
        """
        Parameters
        ----------
        exp_uid : The experiments unique ID.
        query :
        answer:
        butler :

        Returns
        -------
        dictionary with keys:
            alg_args: Keywords that are passed to the algorithm.
            query_update :

        For example, this function might return ``{'a':1, 'b':2}``. The
        algorithm would then be called with
        ``alg.processAnswer(butler, a=1, b=2)``
        """
        #participant_uid = query['participant_uid']
        #participant_uid = args.get('participant_uid', butler.exp_uid)
        participant_uid = butler.queries.get(uid=args['query_uid'],key='participant_uid')
        participant_doc = butler.participants.get(uid=participant_uid)
        #utils.debug_print("participant_doc before increment=", participant_doc)
        butler.participants.increment(uid=participant_uid, key='num_tries')
        #utils.debug_print("participant_doc after increment=", butler.participants.get(uid=participant_uid))
        #if answer['args']['initial_query']:
        if args['initial_query']:
            #initial_arm = answer['args']['answer']['initial_arm']
            initial_arm = args['answer']['initial_arm']
            butler.participants.set(uid=participant_uid, key="i_hat", value=initial_arm)
            alg({'participant_doc': participant_doc})
            return {}
        else:
            #target_id = query['targets'][0]['target']['target_id']
            query_uid = args['query_uid']
            target_id = butler.queries.get(uid=query_uid)['targets'][0]['index']
            #target_reward = answer['args']['answer']['target_reward']
            # utils.debug_print("args: ", args)
            target_reward = args['answer']['target_reward']

            butler.participants.append(uid=participant_uid, key='do_not_ask_list', value=target_id)

            #query_update = {'target_id': target_id, 'target_reward': target_reward}

            #alg_args_dict = {'target_id': target_id,
            #                 'target_reward': target_reward,
            #                 'participant_doc': participant_doc}
        alg({'target_id': target_id, 'target_reward': target_reward, 'participant_doc': participant_doc})
        # return query_update, alg_args_dict
        return {'target_id': target_id, 'target_reward': target_reward}

    #def getModel(self, exp_uid, alg_response, args_dict, butler):
    def getModel(self, butler, alg, args):
        #scores, precisions = alg_response
        scores, precisions = alg()
        ranks = (-numpy.array(scores)).argsort().tolist()
        n = len(scores)
        indexes = numpy.array(range(n))[ranks]
        scores = numpy.array(scores)[ranks]
        precisions = numpy.array(precisions)[ranks]
        ranks = range(n)

        targets = []
        for index in range(n):
            targets.append({'index': indexes[index],
                            'target': self.TargetManager.get_target_item(exp_uid, indexes[index]),
                            'rank': ranks[index],
                            'score': scores[index],
                            'precision': precisions[index]})
        num_reported_answers = butler.experiment.get('num_reported_answers')
        return {'targets': targets, 'num_reported_answers': num_reported_answers}

    def getStats(self, exp_uid, stats_request, dashboard, butler):
        """
        Get statistics to display on the dashboard.
        """
        stat_id = stats_request['args']['stat_id']
        task = stats_request['args']['params'].get('task', None)
        alg_label = stats_request['args']['params'].get('alg_label', None)
        functions = {'api_activity_histogram': dashboard.api_activity_histogram,
                     'compute_duration_multiline_plot': dashboard.compute_duration_multiline_plot,
                     'compute_duration_detailed_stacked_area_plot': dashboard.compute_duration_detailed_stacked_area_plot,
                     'response_time_histogram': dashboard.response_time_histogram,
                     'network_delay_histogram': dashboard.network_delay_histogram,
                     'most_current_ranking': dashboard.most_current_ranking}

        default = [self.app_id, exp_uid]
        args = {'api_activity_histogram': default + [task],
                'compute_duration_multiline_plot': default + [task],
                'compute_duration_detailed_stacked_area_plot': default + [task, alg_label],
                'response_time_histogram': default + [alg_label],
                'network_delay_histogram': default + [alg_label],
                'most_current_ranking': default + [alg_label]}
        return functions[stat_id](*args[stat_id])

