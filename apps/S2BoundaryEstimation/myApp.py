import json
import next.utils as utils
import next.apps.SimpleTargetManager
import networkx as nx
from networkx.readwrite import json_graph

class MyApp:
    def __init__(self,db):
        self.app_id = 'S2BoundaryEstimation'
        self.TargetManager = next.apps.SimpleTargetManager.SimpleTargetManager(db)

    def initExp(self, butler, init_algs, args):
        args['n']  = len(args['targets']['targetset'])

        targets = args['targets']['targetset']
        self.TargetManager.set_targetset(butler.exp_uid, targets)
        del args['targets']

        ### reconstruct & save the graph
        G = _nx_from_neighbors(targets)
        butler.experiment.set(key='G', value=json_graph.adjacency_data(G))

        alg_data = {'n': args['n']}
        init_algs(alg_data)
        return args

    def getQuery(self, butler, alg, args):
        alg_response = alg({'participant_uid':args['participant_uid']})
        target = self.TargetManager.get_target_item(butler.exp_uid, alg_response)
        return {'target_indices':target}

    def processAnswer(self, butler, alg, args):
        query = butler.queries.get(uid=args['query_uid'])
        target = query['target_indices']
        target_label = args['target_label']

        num_reported_answers = butler.experiment.increment(key='num_reported_answers_for_' + query['alg_label'])

        alg({'target_index':target['target_id'],'target_label':target_label})
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
