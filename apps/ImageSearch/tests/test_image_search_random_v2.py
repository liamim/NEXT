import numpy
import numpy.random
import random
import json
import time
from datetime import datetime
import requests
from scipy.linalg import norm
import time
from multiprocessing import Pool
import sys
from scipy.io import loadmat
import numpy as np
import pickle
from sklearn.preprocessing import normalize

import os
HOSTNAME = os.environ.get('NEXT_BACKEND_GLOBAL_HOST', 'localhost')+':'+os.environ.get('NEXT_BACKEND_GLOBAL_PORT', '8000')

TAKE_SIGN = False

def reward(x, theta, R=2):
    r = np.inner(x, theta) + R*np.random.randn()
    if TAKE_SIGN:
        return np.sign(r)
    return r

def run_all(assert_200):
    X = np.load('/Users/scott/Dropbox/Public/gaussian_5_1000.npy')
    X = np.random.randn(2, 1000)
    X = normalize(X, axis=0)
    p, n = X.shape
    i_star = n // 3
    theta_star = X[:, i_star]

    app_id = 'ImageSearch'
    # num_arms = 1000
    # true_means = numpy.array(range(num_arms)[::-1])/float(num_arms)
    total_pulls_per_client = 100

    num_experiments = 1

    # clients run in simultaneous fashion using multiprocessing library
    num_clients = 1

    pool = Pool(processes=num_clients)

    # input test parameters
    delta = 0.05
    supported_alg_ids = ['OFUL']

    labels = [{'label':'yes','reward':1.},{'label':'no','reward':-1.}]

    alg_list = []
    for i, alg_id in enumerate(supported_alg_ids):
        alg_item = {}
        alg_item['alg_id'] = alg_id
        alg_item['alg_label'] = alg_id+'_'+str(i)
        # alg_item['params'] = {}
        alg_list.append(alg_item)
    params = []
    #params['proportions'] = []
    for algorithm in alg_list:
        params.append(  { 'alg_label': algorithm['alg_label'] , 'proportion':1./len(alg_list) }  )
    algorithm_management_settings = {}
    algorithm_management_settings['mode'] = 'fixed_proportions'
    algorithm_management_settings['params'] = params

    print algorithm_management_settings


    #################################################
    # Test POST Experiment
    #################################################
    initExp_args_dict = {}
    initExp_args_dict['args'] = {}

    initExp_args_dict['args']['targets'] = {'n':n}
    initExp_args_dict['args']['features'] = {'matrix': X.tolist()}
    initExp_args_dict['args']['failure_probability'] = delta
    initExp_args_dict['args']['participant_to_algorithm_management'] = 'one_to_many' # 'one_to_one'  #optional field
    initExp_args_dict['args']['algorithm_management_settings'] = algorithm_management_settings #optional field
    initExp_args_dict['args']['alg_list'] = alg_list #optional field
    initExp_args_dict['args']['instructions'] = 'You want instructions, here are your test instructions'
    initExp_args_dict['args']['debrief'] = 'You want a debrief, here is your test debrief'
    initExp_args_dict['args']['context_type'] = 'text'
    initExp_args_dict['args']['context'] = 'This is a context'
    initExp_args_dict['args']['R'] = 2.0
    initExp_args_dict['args']['rating_scale'] = {'labels':labels}
    initExp_args_dict['app_id'] = app_id

    exp_info = []
    for ell in range(num_experiments):
        url = "http://"+HOSTNAME+"/api/experiment"
        response = requests.post(url, json.dumps(initExp_args_dict), headers={'content-type':'application/json'})
        print "POST initExp response =",response.text, response.status_code

        if assert_200: assert response.status_code is 200
        initExp_response_dict = json.loads(response.text)
        if 'fail' in initExp_response_dict['meta']['status'].lower():
              print 'The experiment initialization failed... exiting'
              sys.exit()

        exp_uid = initExp_response_dict['exp_uid']

        exp_info.append( {'exp_uid':exp_uid,} )

        #################################################
        # Test GET Experiment
        #################################################
        url = "http://"+HOSTNAME+"/api/experiment/"+exp_uid
        response = requests.get(url)
        print "GET experiment response =",response.text, response.status_code
        if assert_200: assert response.status_code is 200
        initExp_response_dict = json.loads(response.text)



    ###################################
    # Generate participants and their responses
    ###################################

    participants = []
    pool_args = []
    for i in range(num_clients):
        participant_uid = '%030x' % random.randrange(16**30)
        participants.append(participant_uid)

        experiment = numpy.random.choice(exp_info)
        exp_uid = experiment['exp_uid']
        pool_args.append((exp_uid,participant_uid,total_pulls_per_client,X,i_star,assert_200))

    # generate the responses
    results = map(simulate_one_client, pool_args)

    for i, result in enumerate(results):
        print result[0]
        day = datetime.now().isoformat()[:10]
        filename = 'results/{}/random_{}_{}_{}.pkl'.format(day,
                                    supported_alg_ids,
                                    total_pulls_per_client, i)
        pickle.dump(result[1], open(filename, 'wb'))



def simulate_one_client( input_args ):
    exp_uid,participant_uid,total_pulls,X,i_star,assert_200 = input_args
    avg_response_time = 0.01

    getQuery_times = []
    processAnswer_times = []
    i_hats = []
    for t in range(total_pulls):
        print "    Participant {} had {} total pulls: ".format(participant_uid, t)

        #######################################
        # test POST getQuery #
        #######################################
        getQuery_args_dict = {}
        getQuery_args_dict['exp_uid'] = exp_uid
        getQuery_args_dict['args'] = {}
        # getQuery_args_dict['args']['participant_uid'] = numpy.random.choice(participants)
        getQuery_args_dict['args']['participant_uid'] = participant_uid

        url = 'http://'+HOSTNAME+'/api/experiment/getQuery'
        response,dt = timeit(requests.post)(url, json.dumps(getQuery_args_dict),headers={'content-type':'application/json'})
        # print "POST getQuery response = ", response.text, response.status_code
        print 'POST getQuery response, {}'.format(response.status_code)
        if assert_200: assert response.status_code is 200
        print "POST getQuery duration = ", dt
        getQuery_times.append(dt)
        print

        query_dict = json.loads(response.text)
        query_uid = query_dict['query_uid']
        if t == 0:
            initial_indices = [query_dict['targets'][i]['index']
                                    for i in range(len(query_dict['targets']))]
            i_hat = random.choice(initial_indices)
            i_hat = i_star - 100
            i_hats += [i_hat]
            answer = i_hat
            answer_key = 'initial_arm'
            rewards = [1]
        else:
            # print(query_dict['targets'][0]['index'])
            # targets = query_dict['targets'][0]['index']
            i_hat = query_dict['targets'][0]['index']
            i_hats += [i_hat]
            sqrt = np.sqrt
            # decision = np.inner(X[:, i_hat], theta) + R*np.random.randn()
            print('Pulled arm {} @ iteration {}'.format(i_hat, t))
            # answer = 1 if norm(X[:, i_star] - X[:, i_hat]) < 0.5 / sqrt(1)\
                       # else -1
            answer = reward(X[:, i_hat], X[:, i_star])
            rewards += [answer]
            answer_key = 'target_reward'
        # targets = query_dict['target_indices']
        # target_index = targets[0]['target']['target_id']

        # generate simulated reward #
        #############################
        # sleep for a bit to simulate response time
        ts = time.time()

        # time.sleep(  avg_response_time*numpy.random.rand()  )
        time.sleep(  avg_response_time*numpy.log(1./numpy.random.rand())  )
        # target_reward = true_means[target_index] + numpy.random.randn()*0.5
        # target_reward = 1.+sum(numpy.random.rand(2)<true_means[target_index]) # in {1,2,3}

        # TODO
        target_reward = None
        # target_reward = numpy.random.choice(labels)['reward']

        response_time = time.time() - ts


        #############################################
        # test POST processAnswer 
        #############################################
        processAnswer_args_dict = {}
        processAnswer_args_dict["exp_uid"] = exp_uid
        processAnswer_args_dict["args"] = {}
        processAnswer_args_dict['args']['initial_query'] = True if t==0 else False
        processAnswer_args_dict["args"]["query_uid"] = query_uid
        processAnswer_args_dict["args"]['answer'] = {answer_key: answer}
        processAnswer_args_dict["args"]['response_time'] = response_time

        url = 'http://'+HOSTNAME+'/api/experiment/processAnswer'
        # print "POST processAnswer args = ", processAnswer_args_dict
        response,dt = timeit(requests.post)(url, json.dumps(processAnswer_args_dict), headers={'content-type':'application/json'})
        print "POST processAnswer response", response.text, response.status_code
        if assert_200: assert response.status_code is 200
        print "POST processAnswer duration = ", dt
        processAnswer_times.append(dt)
        print
        processAnswer_json_response = eval(response.text)

    print('here')
    processAnswer_times.sort()
    getQuery_times.sort()
    return_str = '%s \n\t getQuery\t : %f (5),    %f (50),    %f (95)\n\t processAnswer\t : %f (5),    %f (50),    %f (95)\n' % (participant_uid,getQuery_times[int(.05*total_pulls)],getQuery_times[int(.50*total_pulls)],getQuery_times[int(.95*total_pulls)],processAnswer_times[int(.05*total_pulls)],processAnswer_times[int(.50*total_pulls)],processAnswer_times[int(.95*total_pulls)])
    exp_params_to_save = {'i_hats': i_hats,
                          'rewards': rewards,
                          'i_star': i_star,
                          'X': X.tolist()}
    return return_str, exp_params_to_save


def timeit(f):
  """
  Utility used to time the duration of code execution. This script can be composed with any other script.

  Usage::\n
    def f(n): 
      return n**n  

    def g(n): 
      return n,n**n 

    answer0,dt = timeit(f)(3)
    answer1,answer2,dt = timeit(g)(3)
  """
  def timed(*args, **kw):
    ts = time.time()
    result = f(*args, **kw)
    te = time.time()
    if type(result)==tuple:
      return result + ((te-ts),)
    else:
      return result,(te-ts)
  return timed

if __name__ == '__main__':
  print HOSTNAME
  run_all(False)
