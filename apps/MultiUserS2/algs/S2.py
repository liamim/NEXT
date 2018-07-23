# encoding: utf-8

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import itertools
from collections import deque
from next.utils import debug_print, profile_each_line

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
