import numpy
import networkx as nx

class MyAlg:
    def initExp(self, butler, n):
        butler.algorithms.set(key='n',value=n)

        return True

    def getQuery(self, butler, participant_uid):
        # Retrieve the number of targets and return the index of one at random
        n = butler.algorithms.get(key='n')
        idx = numpy.random.choice(n)
        return idx

    def processAnswer(self, butler, target_index, target_label):
        ## TODO: VOTING ##
        setname = butler.exp_uid + {1: '__s_U', -1: '__s_V'}[target_label]
        butler.algorithms.memory.cache.sadd(setname, target_index)

        return True


    def getModel(self, butler):
        return {}
