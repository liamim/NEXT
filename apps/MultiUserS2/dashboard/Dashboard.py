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

