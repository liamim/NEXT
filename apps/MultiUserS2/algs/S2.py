# encoding: utf-8

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import itertools
from collections import deque
from next.utils import debug_print, profile_each_line
import operator
from pprint import pprint

class Status:
    NOT_ASSIGNED = 'not_assigned'
    WAITING      = 'waiting'
    COMPLETED    = 'completed'

class Master(object):
    def __init__(self, exp_uid, n_graphs, job_list, required_votes):
        self.exp_uid = exp_uid
        self.job_list = job_list
        self.required_votes = required_votes
        self.n_graphs = n_graphs

        # one per graph_id
        self.s2_instances = []

    def init_jobs(self):
        pass

    def get_vertex_for(self, user):
        if jobs == []:
            # handle empty job list case
            pass

        opt_job = max([{'job': job, 'priority': priority(job, user, state)} for job in jobs],
                      key=operator.itemgetter('priority'))

        if opt_job['priority'] == -np.inf:
            # handle no assignable jobs case
            pass

class MyAlg:
    def initExp(self, butler, n_graphs, graph_sizes, required_votes):
        # we're just, uh. gonna bypass the Butler, and also DatabaseAPI, because we really should be using
        # a /collection/ to store the priority list (-_-;)
        db = butler.db.client[butler.db.db_name]
        master = Master(butler.exp_uid, n_graphs, db.s2_job_list, required_votes)
        master.init_job_list()

        return True

    def getQuery(self, butler, participant_uid):
        return 0

    def processAnswer(self, butler, target_index, target_label, participant_uid):
        return True

    def getModel(self, butler):
        return {}
