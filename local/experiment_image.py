import os, sys

# import launch_experiment. We assume that it is located in the next-discovery top level directory.
sys.path.append("../")
from launch_experiment import *

experiment_list = []
#alg_ids = ['Greedy']
#alg_ids = ['OFUL']
#alg_ids = ['OFUL_Hashing']
alg_ids = ['OFUL_lite']

# Create common alg_list
alg_list = []
for idx,alg_id in enumerate(alg_ids):
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
algorithm_management_settings['mode'] = 'fixed_proportions'
algorithm_management_settings['params'] = params


# Create experiment dictionary
initExp = {}
initExp['args'] = {}
initExp['args']['d'] = 100
initExp['args']['rating_scale'] = {'labels':[{'label': 'No', 'reward': -1},
                                             {'label': 'Yes', 'reward': 1}]}


#with open(sys.argv[1], 'r') as f:
#    filenames = json.load(f).keys()
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

filenames = [n.keys()[0] for i,n in enumerate(data)]

initExp['args']['feature_filenames'] = filenames
initExp['args']['features'] = 'http://localhost:8002/features_10x10.npy'
initExp['args']['features'] = 'https://www.dropbox.com/s/r9qlkppxvtomk9t/features.npy?dl=1'
initExp['args']['features'] = 'https://www.dropbox.com/s/2sfxmo6pg3yw5d0/features_10x10.npy?dl=1'
initExp['args']['failure_probability'] = .1
initExp['args']['participant_to_algorithm_management'] = 'one_to_many' 
initExp['args']['algorithm_management_settings'] = algorithm_management_settings 
initExp['args']['alg_list'] = alg_list
initExp['R'] = 0.001 #For OFUL 0.001
initExp['ridge'] = 0.1
initExp['args']['instructions'] = 'Test instructions'
initExp['args']['debrief'] = 'Test debrief'
initExp['app_id'] = 'ImageSearch'
#initExp['site_id'] = 'replace this with working site id'
#initExp['site_key'] = 'replace this with working site key'


curr_dir = os.path.dirname(os.path.abspath(__file__))
experiment = {}
experiment['initExp'] = initExp
experiment['primary_type'] = 'json-list'
experiment['primary_target_file'] = sys.argv[1]
experiment_list.append(experiment)

# Launch the experiment
host = "localhost:8000"
print "It's happening"
exp_uid_list = launch_experiment(host, experiment_list)
print "Made experiments {}".format(exp_uid_list)
# Update the cartoon_dueling.html file wit the exp_uid_list and widget_key_list
# with open('strange_fruit_triplet.html','r') as page:
#   print "opended file"
#   page_string = page.read()
#   page_string = page_string.replace("{{exp_uid_list}}", str(exp_uid_list))
#   page_string = page_string.replace("{{widget_key_list}}", str(widget_key_list))
#   with open('../../next_frontend_base/next_frontend_base/templates/strange_fruit_triplet.html','w') as out:
#     out.write(page_string)
#     out.flush()
#     out.close()
