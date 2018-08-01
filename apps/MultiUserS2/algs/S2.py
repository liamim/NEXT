# encoding: utf-8

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import itertools
from collections import deque
from next.utils import debug_print, profile_each_line
import operator
import random
from pprint import pprint

class S2(object):
    def __init__(self, G, gid):
        self.gid = gid
        self.G = G

    def get_query(self):
        if True: # TODO: _active_ learning!
            return random.choice(list(self.G.nodes()))

class Status(object):
    NOT_ASSIGNED = 'not_assigned'
    WAITING      = 'waiting'
    COMPLETED    = 'completed'

class Master(object):
    def __init__(self, butler, db, exp_uid, n_graphs, graph_sizes, required_votes):
        self.butler = butler
        self.db = db
        self.exp_uid = exp_uid
        self.required_votes = required_votes
        self.n_graphs = n_graphs
        self.graph_sizes = graph_sizes

        butler.algorithms.set(key='graph_sizes', value=graph_sizes)

        # reconstitute graphs
        Gs = []
        for gid in range(n_graphs):
            targets_query = self.db.targets.find({'exp_uid': exp_uid, 'graph_id': gid})
            # print(next(targets_query))

            G = nx.Graph()
            for target in targets_query:
                i = target['node_id']
                G.add_node(i)
                for j in target['neighbors']:
                    G.add_edge(i, j)

            Gs.append(G)

        db.graphs.insert_many([{'G': nx.json_graph.node_link_data(G), 'id': gid, 'exp_uid': exp_uid} for gid, G in enumerate(Gs)])

        # one per graph_id
        self.s2_instances = [S2(G, gid) for gid, G in enumerate(Gs)]

    def init_job_list(self):
        jobs = []

        for graph_id, instance in enumerate(self.s2_instances):
            query = instance.get_query()
            jobs.extend([{
                "exp_uid": self.exp_uid,
                "graph_id": graph_id,
                "node_id": query,
                "ballot_id": bid,
                "status": Status.NOT_ASSIGNED
            } for bid in range(self.required_votes)])

        debug_print("computed jobs: {}".format(jobs))

    def get_vertex_for(self, user, priority):
        if self.db.job_list.count({"_id": self.exp_uid}):
            # TODO: handle empty job list case
            return None

        state = None # TODO
        opt_job = max([{'job': job, 'priority': priority(job, user, state)}
                        for job in self.db.job_list.find({"_id": self.exp_uid,
                                                       "status": Status.NOT_ASSIGNED})],
                      key=operator.itemgetter('priority'))

        if opt_job['priority'] == -np.inf:
            # TODO: handle no assignable jobs case
            return None

        return opt_job['job']

class MyAlg:
    def initExp(self, butler, n_graphs, graph_sizes, required_votes):
        # we're just, uh. gonna bypass the Butler, and also DatabaseAPI, because we really should be using
        # a /collection/ to store the priority list (-_-;)
        db = butler.db.client[butler.db.db_name]
        master = Master(butler, db, butler.exp_uid, n_graphs, graph_sizes, required_votes)
        master.init_job_list()

        return True

    def getQuery(self, butler, participant_uid):
        return 0

    def processAnswer(self, butler, target_index, target_label, participant_uid):
        return True

    def getModel(self, butler):
        return {}
