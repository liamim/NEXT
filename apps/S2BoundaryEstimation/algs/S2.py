import numpy

class MyAlg:
    def initExp(self, butler, n):
        # Save the number of targets, dimension, and failure_probability to algorithm storage
        butler.algorithms.set(key='n',value= n)

        return True

    def getQuery(self, butler, participant_uid):
        # Retrieve the number of targets and return the index of one at random
        n = butler.algorithms.get(key='n')
        idx = numpy.random.choice(n)
        return idx

    def processAnswer(self, butler, target_index, target_label):
        return True


    def getModel(self, butler):
        return {}
