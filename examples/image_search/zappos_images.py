"""
Usage:
python chicago_images.py
"""

import os, sys, json
import zipfile
import numpy as np

sys.path.append("../")
from launch_experiment import *

experiment_list = []

# alg_list = [{'alg_id': 'TS', 'alg_label': 'TS'},
#             {'alg_id': 'NN', 'alg_label': 'NN'}]

# alg_list = [{'alg_id': 'Epsilon_Greedy', 'alg_label': 'Epsilon_Greedy'}]
# alg_list = [{'alg_id': 'TS', 'alg_label': 'TS'}]
alg_list = [{'alg_id': 'OFUL', 'alg_label': 'OFUL'}]


# params = [{'alg_label': 'TS', 'proportion':.1},
#           {'alg_label': 'NN', 'proportion':.9}]
# params = [{'alg_label': 'Epsilon_Greedy', 'proportion':1.}]
# params = [{'alg_label': 'TS', 'proportion':1.}]
params = [{'alg_label': 'OFUL', 'proportion':1.}]

initExp = {}
initExp['args'] = {}
initExp['args']['d'] = 1000
initExp['args']['failure_probability'] = .1
initExp['args']['participant_to_algorithm_management'] = 'one_to_one'
initExp['args']['algorithm_management_settings'] = {'mode':'fixed_proportions', 'params': params}
initExp['args']['alg_list'] = alg_list
initExp['args']['rating_scale'] = {'labels': [{'label': 'No', 'reward': -1},
                                              {'label': 'Yes', 'reward': 1}]}

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

filenames = [n.keys()[0] for i, n in enumerate(data)]

initExp['args']['features'] = 'not needed'
initExp['args']['feature_filenames'] = filenames
#initExp['args']['instructions'] = ("Please select, using your mouse or left and right arrow keys, the item on the bottom that is closest to the top.<br/>"
#                                   "<h2> Which of the two places on the bottom looks like the place on the top?</h2>")
initExp['args']['num_tries'] = 50

initExp['args'][
    'instructions'] = ' '
initExp['args']['debrief'] = 'Thanks for answering questions. You can close the window now.'
initExp['app_id'] = 'ImageSearch'

curr_dir = 'os.path.dirname(os.path.abspath(__file__))'
experiment = {}
experiment['initExp'] = initExp

# The user chooses between two images. This could be text or video as well.
experiment['primary_type'] = 'image'
experiment['primary_target_file'] = 'zappos_50k_dropbox.json'
# experiment['primary_target_file'] = 'zappos_10_dropbox.json'
experiment_list.append(experiment)

# Launch the experiment.
try:
  AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
  AWS_ACCESS_ID = os.environ['AWS_ACCESS_KEY_ID']
  AWS_BUCKET_NAME = os.environ['AWS_BUCKET_NAME']
  host = os.environ['NEXT_BACKEND_GLOBAL_HOST']+ ":" \
          + os.environ.get('NEXT_BACKEND_GLOBAL_PORT', '8000')
  exp_uid_list = launch_experiment(host, experiment_list, AWS_ACCESS_ID,
                                   AWS_SECRET_ACCESS_KEY, AWS_BUCKET_NAME,
                                   parallel_upload=True)

except:
    print 'No AWS defined, luaching without S3'
    host = "localhost:8000"

    exp_uid_list = launch_experiment_noS3(host, experiment_list)

# Call launch_experiment module found in NEXT/lauch_experiment.py
