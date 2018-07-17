# encoding: utf-8

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import itertools
from next.utils import debug_print

class MyAlg:
    def initExp(self, butler):
        return True

    def getQuery(self, butler, participant_uid):
        return 0

    def processAnswer(self, butler, target_index, target_label, participant_uid):
        return True

    def getModel(self, butler):
        return {}
