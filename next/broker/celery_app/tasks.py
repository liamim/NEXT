from __future__ import absolute_import
from .celery_broker import app
import celery.signals
import os
import sys
import time
import json
import traceback
import numpy
from next.constants import DEBUG_ON
import hashlib
import cPickle as pickle

# import next.logging_client.LoggerHTTP as ell
from next.database_client.DatabaseAPI import DatabaseAPI
db = DatabaseAPI()
from next.logging_client.LoggerAPI import LoggerAPI
ell = LoggerAPI()
import next.utils
import next.constants
import next.apps.Butler as Butler
import next.lib.pijemont.verifier as verifier

Butler = Butler.Butler

class App_Wrapper:
    def __init__(self, app_id, exp_uid, db, ell):
        self.app_id = app_id
        self.exp_uid = exp_uid
        self.next_app = next.utils.get_app(app_id, exp_uid, db, ell)
        self.butler = Butler(app_id, exp_uid, self.next_app.myApp.TargetManager, db, ell)

    def getModel(self, args_in_json):
        response, dt = next.utils.timeit(self.next_app.getModel)(self.next_app.exp_uid, args_in_json)
        args_out_json,didSucceed,message = response
        args_out_dict = json.loads(args_out_json)
        meta = args_out_dict.get('meta',{})
        if 'log_entry_durations' in meta.keys():
            self.log_entry_durations = meta['log_entry_durations']
            self.log_entry_durations['timestamp'] = next.utils.datetimeNow()
        return args_out_dict['args']

# Main application task
def apply(app_id, exp_uid, task_name, args_in_json, enqueue_timestamp):
    enqueue_datetime = next.utils.str2datetime(enqueue_timestamp)
    dequeue_datetime = next.utils.datetimeNow()
    delta_datetime = dequeue_datetime - enqueue_datetime
    time_enqueued = delta_datetime.seconds + delta_datetime.microseconds/1000000.

    # modify args_in
    if task_name == 'processAnswer':
        args_in_dict = json.loads(args_in_json)
        args_in_dict['args']['timestamp_answer_received'] = enqueue_timestamp
        args_in_json = json.dumps(args_in_dict)
    # get stateless app
    next_app = next.utils.get_app(app_id, exp_uid, db, ell)
    # pass it to a method
    method = getattr(next_app, task_name)
    print("ASD",method,next_app)
    response, dt = next.utils.timeit(method)(exp_uid, args_in_json)
    args_out_json,didSucceed,message = response
    args_out_dict = json.loads(args_out_json)
    if 'args' in args_out_dict:
        return_value = (json.dumps(args_out_dict['args']),didSucceed,message)
        meta = args_out_dict.get('meta',{})
        if 'log_entry_durations' in meta:
            log_entry_durations = meta['log_entry_durations']
            log_entry_durations['app_duration'] = dt
            log_entry_durations['duration_enqueued'] = time_enqueued
            log_entry_durations['timestamp'] = next.utils.datetimeNow()
            ell.log( app_id+':ALG-DURATION', log_entry_durations  )
    else:
        return_value = (args_out_json,didSucceed,message)
    print '#### Finished %s,  time_enqueued=%s,  execution_time=%s ####' % (task_name,time_enqueued,dt)
    return return_value

def apply_dashboard(app_id, exp_uid, args_in_json, enqueue_timestamp):
    enqueue_datetime = next.utils.str2datetime(enqueue_timestamp)
    dequeue_datetime = next.utils.datetimeNow()
    delta_datetime = dequeue_datetime - enqueue_datetime
    time_enqueued = delta_datetime.seconds + delta_datetime.microseconds/1000000.
    dir, _ = os.path.split(__file__)
    reference_dict,errs = verifier.load_doc('{}/{}.yaml'.format(app_id, app_id),"apps/")
    if len(errs) > 0:
        raise Exception("App YAML format errors: \n{}".format(str(errs)))
    args_dict = verifier.verify(args_in_json, reference_dict['getStats']['args'])
    stat_id = args_dict['args'].get('stat_id','none')

    stat_args = args_dict['args']

    hash_object = hashlib.md5(stat_id+'_'+json.dumps(stat_args['params']))
    stat_uid = hash_object.hexdigest()
    stat_uid += '_' + exp_uid

    app = App_Wrapper(app_id, exp_uid, db, ell)
    cached_doc = app.butler.dashboard.get(uid=stat_uid)
    cached_response = None
    if (int(stat_args.get('force_recompute',0))==0) and (cached_doc is not None):
        delta_datetime = (next.utils.datetimeNow() - next.utils.str2datetime(cached_doc['timestamp']))
        if delta_datetime.seconds < next.constants.DASHBOARD_STALENESS_IN_SECONDS:
            cached_response = json.loads(cached_doc['data_dict'])
            if 'meta' not in cached_response:
                cached_response['meta']={}
            cached_response['meta']['cached'] = 1
            if delta_datetime.seconds/60<1:
                cached_response['meta']['last_dashboard_update'] = '<1 minute ago'
            else:
                cached_response['meta']['last_dashboard_update'] = str(delta_datetime.seconds/60)+' minutes ago'

    if cached_response==None:
        dashboard_string = 'apps.' + app_id + '.dashboard.Dashboard'
        dashboard_module = __import__(dashboard_string, fromlist=[''])
        dashboard = getattr(dashboard_module, app_id+'Dashboard')
        dashboard = dashboard(db, ell)
        stats_method = getattr(dashboard, stat_id)
        response,dt = next.utils.timeit(stats_method)(app,app.butler,**args_dict['args']['params'])

        save_dict = {'exp_uid':app.exp_uid,
            'stat_uid':stat_uid,
            'timestamp':next.utils.datetime2str(next.utils.datetimeNow()),
            'data_dict':json.dumps(response)}
        app.butler.dashboard.set_many(uid=stat_uid,key_value_dict=save_dict)

        # update the admin timing with the timing of a getModel
        if hasattr(app, 'log_entry_durations'):
            app.log_entry_durations['app_duration'] = dt
            app.log_entry_durations['duration_enqueued'] = time_enqueued
            app.butler.ell.log(app.app_id+':ALG-DURATION', app.log_entry_durations)
    else:
        response = cached_response

    if DEBUG_ON:
        next.utils.debug_print('#### Finished Dashboard %s, time_enqueued=%s,  execution_time=%s ####' % (stat_id, time_enqueued, dt), color='white')
    return json.dumps(response), True, ''


# class HashHelper(celery.Task):
#     # def __init__(self):
#     next.utils.debug_print('UAYSGCUYASGCUYASGCUYASGCUYSAGCUAYSGCUYASCG initializing hash in tasks, pid=%d'%(os.getpid()))
#         #self.lsh = self._get_hashing_function()
#     _lsh = None
#     abstract = True

#     @property
#     def lsh(self):
#         if self._lsh == None:
#             next.utils.debug_print('HAGFSCDGFASCD loading hash for the first time with pid = %d'%(os.getpid()))
#             self._lsh = self._get_hashing_function()
#         else:
#             next.utils.debug_print('AAAAAAAA already loaded for pid = %d' % (os.getpid()))
#         return self._lsh

#     @staticmethod
#     def _get_hashing_function():
#         #with open('hashing_functions_d1000.pkl') as f:
#         with open('hashing_functions.pkl') as f:
#             index = pickle.load(f)

#         return index

# @app.task(base=HashHelper)
# def Hash():
#     return Hash.lsh

# class FeaturesHelper(celery.Task):
#     # def __init__(self):
#     next.utils.debug_print('UAYSGCUYASGCUYASGCUYASGCUYSAGCUAYSGCUYASCG initializing hash in tasks, pid=%d'%(os.getpid()))
#         #self.lsh = self._get_hashing_function()
#     _features = None
#     abstract = True

#     @property
#     def features(self):
#         if self._features == None:
#             next.utils.debug_print('asdasdaHAGFSCDGFASCD loading features for the first time with pid = %d'%(os.getpid()))
#             self._features = self._get_feature_vectors()
#             next.utils.debug_print('done loading features')
#         else:
#             next.utils.debug_print('AasdasdaAAAAAAA already loaded for pid = %d' % (os.getpid()))

#         next.utils.debug_print('serializing features')
#         x = self._features.tolist()
#         next.utils.debug_print('done serializing features')

#         return x

#     @staticmethod
#     def _get_feature_vectors():
#         return numpy.load('features_d1000.npy')


# @app.task(base=FeaturesHelper)
# def Features():
#     return Features.features


# class Features(celery.Task):
#     def __init__(self):
#         print('initializing features in tasks, pid=%d'%(os.getpid()))
        # self.features = self._get_feature_vectors()
        # self.features = None
    #
    # @staticmethod
    # def _get_feature_vectors():
    #     features = numpy.load('features_d1000.npy')
    #     return features
    #
    # def run(self):
    #     if self.features == None:
    #         print('loading features for the first time with pid = %d'%(os.getpid()))
    #         self.features = self._get_feature_vectors()
    #     else:
    #         print('already loaded for pid = %d' % (os.getpid()))
    #     return self.features


# class Hash(celery.Task):
#     print('loading hash for %d...'%(os.getpid()))
#     with open('hashing_functions.pkl') as f:
#         lsh = pickle.load(f)
#
#     print ('done')
#
#     def run(self):
#         print('pid is: ', os.getpid())
#         return self.lsh
#
#
# class Features(celery.Task):
#     print('loading features for %d ...'%(os.getpid()))
#     _features = numpy.load('features_d1000.npy')
#     print ('done')
#
#     def run(self):
#         print('pid is: ', os.getpid())
#         return self._features

def apply_sync_by_namespace(app_id, exp_uid, alg_id, alg_label, task_name, args, namespace, job_uid, enqueue_timestamp, time_limit):
    enqueue_datetime = next.utils.str2datetime(enqueue_timestamp)
    dequeue_datetime = next.utils.datetimeNow()
    delta_datetime = dequeue_datetime - enqueue_datetime
    time_enqueued = delta_datetime.seconds + delta_datetime.microseconds/1000000.

    try:
        print '>>>>>>>> Starting namespace:%s,  job_uid=%s,  time_enqueued=%s <<<<<<<<<' % (namespace,job_uid,time_enqueued)
        # get stateless app
        next_app = next.utils.get_app(app_id, exp_uid, db, ell)
        target_manager = next_app.myApp.TargetManager
        next_alg = next.utils.get_app_alg(app_id, alg_id)
        butler = Butler(app_id, exp_uid, target_manager, db, ell, alg_label, alg_id)
        response,dt = next.utils.timeit(getattr(next_alg, task_name))(butler, args)
        log_entry_durations = { 'exp_uid':exp_uid,'alg_label':alg_label,'task':'daemonProcess','duration':dt }
        log_entry_durations.update(butler.algorithms.getDurations())
        log_entry_durations['app_duration'] = dt
        log_entry_durations['duration_enqueued'] = time_enqueued
        log_entry_durations['timestamp'] = next.utils.datetimeNow()
        ell.log( app_id+':ALG-DURATION', log_entry_durations)
        print '########## Finished namespace:%s,  job_uid=%s,  time_enqueued=%s,  execution_time=%s ##########' % (namespace,job_uid,time_enqueued,dt)
        return
    except Exception, error:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print "tasks Exception: {} {}".format(error, traceback.format_exc())
        traceback.print_tb(exc_traceback)

        # error = traceback.format_exc()
        # log_entry = { 'exp_uid':exp_uid,'task':'daemonProcess','error':error,'timestamp':next.utils.datetimeNow() }
        # ell.log( app_id+':APP-EXCEPTION', log_entry  )
        return None

# forces each worker to get its own random seed.
@celery.signals.worker_process_init.connect()
def seed_rng(**_):
    """
    Seeds the numpy random number generator.
    """
    numpy.random.seed()

# If celery isn't off, celery-wrap the functions so they can be called with apply_async
if next.constants.CELERY_ON:
    apply = app.task(apply)
    apply_dashboard = app.task(apply_dashboard)
    #Hash = app.task(Hash, base=HashHelper)
    #Features = app.task(Features)
    apply_sync_by_namespace = app.task(apply_sync_by_namespace)

