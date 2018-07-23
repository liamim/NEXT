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
        # i would prefer to_dict_of_lists, but that doesn't store node attrs!
        butler.experiment.set(key='G', value=json_graph.node_link_data(G))

        alg_data = {'n': args['n'], 'budget': args['budget'], 'early_stop': args['early_stop']}
        init_algs(alg_data)
        return args

    @profile_each_line
    def getQuery(self, butler, alg, args):
        participant_uid = args['participant_uid']
        user_graph_dat = butler.participants.get(uid=participant_uid, key='G')
        if user_graph_dat is None:
            print("No graph! Loading experiment-wide one.")
            graph_dat = butler.experiment.get(key='G')
            butler.participants.set(uid=participant_uid, key='G', value=graph_dat)
            G = json_graph.node_link_graph(graph_dat)
        else:
            G = json_graph.node_link_graph(user_graph_dat)

        alg_response = alg({'participant_uid':participant_uid})

        # draw it
        fig = plt.figure(figsize=(10, 10), frameon=False)
        ax = fig.add_subplot(111)
        def oracle(n):
            return G.nodes.data('label')[n]
        def pos(n):
            targ = self.TargetManager.get_target_item(butler.exp_uid, n)
            x, y = targ['location']
            return x, y
        draw_labeled_graph(G, oracle, pos, ax=ax)
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(tight=True)
        ax.set_frame_on(False)

        img = StringIO()
        fig.savefig(img, format='png', aspect='normal', bbox_inches='tight', pad_inches=0)
        img_data = base64.encodestring(img.getvalue())

        if alg_response is not None:
            target = self.TargetManager.get_target_item(butler.exp_uid, alg_response)
            return {'target_indices':target, 'graph_img_data': img_data, 'done': False}
        else:
            return {'graph_img_data': img_data, 'done': True}


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

def draw_labeled_graph(G, oracle, pos, ax=None):
    def label_to_color(l):
        if l is None: return 'grey'
        return 'r' if l > 0 else 'b'

    nx.draw(G,
        pos={n: pos(n) for n in G.nodes()},
        node_color=[label_to_color(oracle(n)) for n in G.nodes()],
        ax=ax)
