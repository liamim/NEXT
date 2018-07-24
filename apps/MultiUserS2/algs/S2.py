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

class Master:
    def __init__(self):
        pass

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
    def initExp(self, butler, n):
        butler.algorithms.set(key='n', value=n)

        return True

    def getQuery(self, butler, participant_uid):
        return 0

    def processAnswer(self, butler, target_index, target_label, participant_uid):
        return True

    def getModel(self, butler):
        return {}
