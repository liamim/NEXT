import matplotlib
matplotlib.use('Agg')

import json
from next.utils import profile_each_line, debug_print
import next.apps.SimpleTargetManager
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from cStringIO import StringIO
import base64
import collections
import itertools
import __builtin__


class MyApp:
    def __init__(self,db):
        self.app_id = 'S2BoundaryEstimation'
        self.TargetManager = next.apps.SimpleTargetManager.SimpleTargetManager(db)

    def initExp(self, butler, init_algs, args):
        args['n']  = len(args['targets']['targetset'])

        targets = args['targets']['targetset']
        self.TargetManager.set_targetset(butler.exp_uid, targets)
        del args['targets']

        ### reconstruct & save the graphs
        new_ids = itertools.count()
        id_map = collections.defaultdict(lambda: __builtin__.next(new_ids)) # thanks for naming the root module `next`
        Gs = {}
        for target in targets:
            # print(target)
            gid = id_map[target['graph_id']]
            if Gs.get(gid) is None:
                Gs[gid] = nx.Graph()

            i = target['node_id']
            Gs[gid].add_node(i)

            for j in target['neighbors']:
                Gs[gid].add_edge(i, j)

        for gid, graph in Gs.items():
            butler.experiment.set(key='G_{}'.format(gid), value=json_graph.node_link_data(graph))

        alg_data = {'n_graphs': len(Gs), 'graph_sizes': [G.order() for _, G in Gs.items()], 'required_votes': args['required_votes']}
        init_algs(alg_data)
        return args

    def getQuery(self, butler, alg, args):
        participant_uid = args['participant_uid']
        alg_response = alg({'participant_uid':participant_uid})
        target = self.TargetManager.get_target_item(butler.exp_uid, alg_response)

        return {'target_indices':target}

    def processAnswer(self, butler, alg, args):
        query = butler.queries.get(uid=args['query_uid'])
        target = query['target_indices']
        target_label = args['target_label']
        participant_uid = query['participant_uid']

        num_reported_answers = butler.experiment.increment(key='num_reported_answers_for_' + query['alg_label'])

        alg({'target_index':target['target_id'],'target_label':target_label, 'participant_uid': participant_uid})

        return {'target_index':target['target_id'],'target_label':target_label}

    def getModel(self, butler, alg, args):
        return alg()


def _nx_from_neighbors(targets):
    G = nx.Graph()
    for i, target in enumerate(targets):
        G.add_node(i)
        for j in target['neighbors']:
            G.add_edge(i, j)

    return G
