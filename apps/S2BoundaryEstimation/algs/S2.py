# encoding: utf-8

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import itertools
from next.utils import debug_print

class MyAlg:
    def initExp(self, butler, n, query_repeats):
        butler.algorithms.set(key='n',value=n)
        # TODO: make sure repeats are assigned to different users
        butler.algorithms.set(key='required_voters', value=query_repeats)

        butler.algorithms.set(key='T', value=np.zeros(n))
        butler.algorithms.set(key='X', value=np.zeros(n))

        return True

    def getQuery(self, butler, participant_uid):
        butler.algorithms.memory.ensure_connection() # hhah

        # load the graph
        G = json_graph.node_link_graph(butler.experiment.get(key='G'))
        U = {int(v) for v in butler.algorithms.memory.cache.smembers(butler.exp_uid + '__s_U')}
        V = {int(v) for v in butler.algorithms.memory.cache.smembers(butler.exp_uid + '__s_V')}

        # what vertex we consider next
        idx = find_moss(G, U, V)
        if idx is not None:
            idx = int(idx)

        debug_print("MOSS vert ::: {}".format(idx))

        if idx is None:
            n = butler.algorithms.get(key='n')
            idx = np.random.choice(n)

        return idx

    def processAnswer(self, butler, target_index, target_label):
        butler.algorithms.memory.ensure_connection() # hhah

        X = butler.algorithms.get(key='X')
        T = butler.algorithms.get(key='T')
        X[target_index] += target_label
        T[target_index] += 1
        butler.algorithms.set(key='X', value=X)
        butler.algorithms.set(key='T', value=T)

        # load the graph
        G = json_graph.node_link_graph(butler.experiment.get(key='G'))

        if T[target_index] >= butler.algorithms.get(key='required_voters'):
            y = np.sign(X[target_index])
            setname = butler.exp_uid + {1: '__s_U', -1: '__s_V'}[y]
            butler.algorithms.memory.cache.sadd(setname, target_index)
            G.nodes[target_index]['label'] = y

        # find obvious cuts
        cuts = find_obvious_cuts(G)
        debug_print("Found cuts: {}".format(cuts))
        # unzip
        G.remove_edges_from(cuts)

        # save the graph back
        butler.experiment.set(key='G', value=json_graph.node_link_data(G))

        return True

    def getModel(self, butler):
        return {}

def find_obvious_cuts(G, L=None):
    """
    Find obvious cuts between adjacent verts of different labels.

    Parameters
    ----------
    G : nx.Graph
        The input graph, with nodes with known label marked with the data attribute `label`.
    L : optional list of (vert, label)
        A list of tuples of vertices with known labels, and their corresponding label.

        Is only used to accelerate slightly so we don't have to re-search the graph; if None,
        we do our own search for `label`s.
    """

    if L is None:
        labeled_nodes = [v[0] for v in G.nodes(data=True) if v[1].get('label') is not None]
    else:
        labeled_nodes = [l[0] for l in L]

    labeled_subgraph = G.subgraph(labeled_nodes)

    cuts = []
    # for every pair of labeled vertices
    for edge in labeled_subgraph.edges():
        # for every cut pair of labels
        if G.node[edge[0]]['label'] != G.node[edge[1]]['label']:
            cuts.append(edge)

    return cuts

def find_moss(G, U, V):
    # TODO: replace enumerate_find_ssp with the accelerated ball-growth MOSS algorithm.
    return path_midpoint(enumerate_find_ssp(G, U, V))

def path_midpoint(path):
    if path is None:
        return None

    return path[len(path)//2]

def enumerate_find_ssp(G, U, V):
    """
    Finds all shortest paths between pairs of vertices spanning U and V, and returns the shortest one.

    Time complexity is something like
        O(|U|*|V| * n log n) ≈ O((n/2)^2 * n log n) = O(n^3 log n)
    where n = |U| + |V|.

    Probably don't use this, _especially_ if you have a lattice graph.
    """

    paths = []

    # for every pair of labeled vertices
    for (u, v) in itertools.product(U, V):
        try:
            P = nx.shortest_path(G, u, v)
        except nx.NetworkXNoPath:
            # we don't have a path, so skip adding it to the list
            continue

        paths.append(P)

    # if we found no paths, return None
    if paths == []:
        return None

    # find the shortest shortest path
    ssp = min(paths, key=lambda path: len(path))

    return ssp
