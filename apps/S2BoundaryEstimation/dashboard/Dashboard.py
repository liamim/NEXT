import json
import numpy
import numpy.random
from datetime import datetime
from datetime import timedelta
import next.utils as utils
from next.apps.AppDashboard import AppDashboard
import matplotlib.pyplot as plt
import mpld3
import networkx as nx
from networkx.readwrite import json_graph

class MyAppDashboard(AppDashboard):
    def __init__(self,db,ell):
        AppDashboard.__init__(self, db, ell)

    def current_graph_state_figure(self, app, butler):
        # load the graph
        G = json_graph.node_link_graph(butler.experiment.get(key='G'))

        # draw it
        fig = plt.figure(figsize=(10, 10), frameon=False)
        ax = fig.add_subplot(111)
        def oracle(n):
            return G.nodes.data('label')[n]
        def pos(n):
            targ = butler.targets.get_target_item(butler.exp_uid, n)
            x, y = targ['location']
            return x, y
        draw_labeled_graph(G, oracle, pos, ax=ax)
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(tight=True)
        ax.set_frame_on(False)

        plot_dict = mpld3.fig_to_dict(fig)
        plt.close()
        return plot_dict

def draw_labeled_graph(G, oracle, pos, ax=None):
    def label_to_color(l):
        if l is None: return 'grey'
        return 'r' if l > 0 else 'b'

    nx.draw(G,
        pos={n: pos(n) for n in G.nodes()},
        node_color=[label_to_color(oracle(n)) for n in G.nodes()],
        ax=ax)
