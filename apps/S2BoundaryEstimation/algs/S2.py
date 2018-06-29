# encoding: utf-8

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import itertools
from next.utils import debug_print

class MyAlg:
    def initExp(self, butler, n):
        butler.algorithms.set(key='n',value=n)
        butler.algorithms.set(key='required_voters', value=1)

        butler.algorithms.set(key='T', value=np.zeros(n))
        butler.algorithms.set(key='X', value=np.zeros(n))

        return True

    def getQuery(self, butler, participant_uid):
        # Retrieve the number of targets and return the index of one at random
        n = butler.algorithms.get(key='n')
        idx = numpy.random.choice(n)
        return idx

    def processAnswer(self, butler, target_index, target_label):
        # load the graph, somehow?
        # graph = _load_graph

        ## TODO: VOTING ##
        setname = butler.exp_uid + {1: '__s_U', -1: '__s_V'}[target_label]
        butler.algorithms.memory.cache.sadd(setname, target_index)

        return True

    def getModel(self, butler):
        return {}
