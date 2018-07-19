# encoding: utf-8

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import itertools
from collections import deque
from next.utils import debug_print, profile_each_line

class MyAlg:
    def initExp(self, butler, n, budget):
        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='budget', value=budget)

        return True

    @profile_each_line
    def getQuery(self, butler, participant_uid):
        user_graph_dat = butler.participants.get(uid=participant_uid, key='G')
        if user_graph_dat is None:
            print("No graph! Loading experiment-wide one.")
            graph_dat = butler.experiment.get(key='G')
            butler.participants.set(uid=participant_uid, key='G', value=graph_dat)
            G = json_graph.node_link_graph(graph_dat)
        else:
            G = json_graph.node_link_graph(user_graph_dat)

        U = {int(v) for v in butler.participants.get(uid=participant_uid, key='items_U') or []}
        V = {int(v) for v in butler.participants.get(uid=participant_uid, key='items_V') or []}

        # what vertex we consider next
        idx = find_moss(G, U, V)
        debug_print("MOSS vert ::: {}".format(idx))
        if not butler.participants.get(uid=participant_uid, key='enough_random_samples'):
            if idx is None:
                n = butler.algorithms.get(key='n')
                idx = np.random.choice(n)
            else:
                butler.participants.set(uid=participant_uid, key='enough_random_samples', value=True)
        else:
            if idx is None:
                # we're done. we separated the components.
                return None

        if butler.participants.get(uid=participant_uid, key='n_responses') > butler.algorithms.get(key='budget'):
            return None

        debug_print("choosing vert ::: {}".format(idx))

        return idx

    def processAnswer(self, butler, target_index, target_label, participant_uid):
        butler.participants.increment(uid=participant_uid, key='n_responses')

        # load the graph
        G = json_graph.node_link_graph(butler.participants.get(uid=participant_uid, key='G'))

        y = int(target_label)
        keyname = {1: 'items_U', -1: 'items_V'}[y]
        butler.participants.append(uid=participant_uid, key=keyname, value=target_index)
        G.nodes[target_index]['label'] = y

        # find obvious cuts
        cuts = find_obvious_cuts(G)
        debug_print("Found cuts: {}".format(cuts))
        # unzip
        G.remove_edges_from(cuts)

        # save the graph back
        butler.participants.set(uid=participant_uid, key='G', value=json_graph.node_link_data(G))

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
    return accel_moss(G, U, V)

def path_midpoint(path):
    if path is None:
        return None

    return path[len(path)//2]

def accel_moss(G, U, V):
    queue_u, queue_v = deque([]), deque([])
    visited_u, visited_v = U.copy(), V.copy()

    for u in U:
        queue_u.append((u, G.neighbors(u)))

    for v in V:
        queue_v.append((v, G.neighbors(v)))

    while queue_u and queue_v:
        parent, children = queue_u.popleft()
        for child in children:
            if child not in visited_u:
                visited_u.add(child)
                queue_u.append((child, G.neighbors(child)))
                if child in visited_v and child not in V:
                    return child

        parent, children = queue_v.popleft()
        for child in children:
            if child not in visited_v:
                visited_v.add(child)
                queue_v.append((child, G.neighbors(child)))
                if child in visited_u and child not in U:
                    return child

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
