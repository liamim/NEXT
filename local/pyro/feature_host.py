# saved as greeting-server.py
import Pyro4
import numpy as np
import cPickle as pickle
path = '../../'
import sys
sys.path.insert(0, path)
import lsh_kjun_v3

@Pyro4.expose
class DataHost(object):
    def __init__(self):
        self.X = self._get_feature_vectors()
        print('done')
        self.lsh = self._get_hashing_function()

    def _get_hashing_function(self):
        # try:
        #    with open('hashing_functions.pkl') as f:
        #        data = pickle.load(f)
        # except:
        #    raise ValueError('Current path:', os.getcwd())
        # from next.lib.hash import kjunutils, lsh_kjun_v3`
        print('Loading hash function ...')
        # with open('../../hashing_functions.pkl') as f:
        #     index = pickle.load(f)
        # with open('hashing_functions_d1000.pkl') as f:
        #     index = pickle.load(f)
        print('done')
        index = []
        #index = hash.to_serializable(index)
        return index

    def _get_feature_vectors(self):
        # features = np.load('features_10x10.npy')
        print('Loading features function ...')
        return np.load('../../features_d1000.npy').tolist()

    def get_features(self):
        return self.X

    def get_hash(self):
        return self.lsh


daemon = Pyro4.Daemon()                # make a Pyro daemon
ns = Pyro4.locateNS()                  # find the name server
uri = daemon.register(DataHost)   # register the greeting maker as a Pyro object
ns.register("data.host", uri)   # register the object with a name in the name server

print("Ready.")
daemon.requestLoop()                   # start the event loop of the server to wait for calls