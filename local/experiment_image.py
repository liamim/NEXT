import os, sys

# import launch_experiment. We assume that it is located in the next-discovery top level directory.
sys.path.append("../")
from launch_experiment import *

experiment_list = []
# alg_ids = ['Greedy']
# alg_ids = ['OFUL']
# alg_ids = ['OFUL_Hashing']
# alg_ids = ['OFUL_lazy_lsh']
# alg_ids = ['OFUL_lite']
# alg_ids = ['TS']
# alg_ids = ['NN']
alg_ids = ['OFUL_Hashing', 'OFUL_lite', 'TS', 'OFUL_lazy_lsh', 'NN']

# Create common alg_list
alg_list = []
for idx, alg_id in enumerate(alg_ids):
    alg_item = {}
    alg_item['alg_id'] = alg_id
    alg_item['alg_label'] = alg_id
    alg_list.append(alg_item)

# Create common algorithm management settings
params = []
for algorithm in alg_list:
    params += [{'alg_label': algorithm['alg_label'],
                'proportion': 1.0 / len(alg_list)}]

algorithm_management_settings = {}
# algorithm_management_settings['mode'] = 'fixed_proportions'
algorithm_management_settings['mode'] = 'custom'
algorithm_management_settings['params'] = params

# Create experiment dictionary
initExp = {}
initExp['args'] = {}
initExp['args']['rating_scale'] = {'labels': [{'label': 'No', 'reward': -1},
                                              {'label': 'Yes', 'reward': 1}]}

# with open(sys.argv[1], 'r') as f:
#    filenames = json.load(f).keys()
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

filenames = [n.keys()[0] for i, n in enumerate(data)]

initExp['args']['features'] = 'not needed'
initExp['args']['feature_filenames'] = filenames
initExp['args']['failure_probability'] = .1
initExp['args']['participant_to_algorithm_management'] = 'one_to_one'
initExp['args']['algorithm_management_settings'] = algorithm_management_settings
initExp['args']['alg_list'] = alg_list
initExp['R'] = 0.001  # For OFUL 0.001
initExp['ridge'] = 0.1
initExp['args'][
    'instructions'] = ' '
initExp['args']['debrief'] = 'Thanks for answering questions. You can close the window now.'
initExp['app_id'] = 'ImageSearch'


curr_dir = os.path.dirname(os.path.abspath(__file__))
experiment = {}
experiment['initExp'] = initExp
experiment['primary_type'] = 'json-list'
experiment['primary_target_file'] = sys.argv[1]
experiment_list.append(experiment)

# Launch the experiment
# host = "localhost:8000"
host = 'ec2-35-167-181-81.us-west-2.compute.amazonaws.com:8000'
# host = 'next.discovery.wisc.edu'
print "It's happening"
exp_uid_list = launch_experiment(host, experiment_list)
print "Made experiments {}".format(exp_uid_list)
