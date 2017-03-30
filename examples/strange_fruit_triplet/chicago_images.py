"""
Usage:
python chicago_images.py
"""

import os, sys
import zipfile
import numpy as np

sys.path.append("../")
from launch_experiment import *

experiment_list = []

alg_list = [{'alg_id': 'RandomSampling', 'alg_label': 'Random', 'test_alg_label':'Test'},
            {'alg_id': 'RandomSampling', 'alg_label': 'Test', 'test_alg_label':'Test'}]

params = [{'alg_label': 'Test', 'proportion':.1},
          {'alg_label': 'Random', 'proportion':.9}]

initExp = {}
initExp['args'] = {}
initExp['args']['d'] = 2
initExp['args']['failure_probability'] = .01
initExp['args']['participant_to_algorithm_management'] = 'one_to_many'
initExp['args']['algorithm_management_settings'] = {'mode':'fixed_proportions', 'params': params}
initExp['args']['alg_list'] = alg_list
#initExp['args']['instructions'] = ("Please select, using your mouse or left and right arrow keys, the item on the bottom that is closest to the top.<br/>"
#                                   "<h2> Which of the two places on the bottom looks like the place on the top?</h2>")
initExp['args']['num_tries'] = 50 


initExp['app_id'] = 'PoolBasedTripletMDS'

curr_dir = 'os.path.dirname(os.path.abspath(__file__))'
experiment = {}
experiment['initExp'] = initExp

# The user chooses between two images. This could be text or video as well.
experiment['primary_type'] = 'image'
experiment['primary_target_file'] = 'test.json'
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
