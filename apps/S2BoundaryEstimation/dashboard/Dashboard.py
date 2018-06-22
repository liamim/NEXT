import json
import numpy
import numpy.random
from datetime import datetime
from datetime import timedelta
import next.utils as utils
from next.apps.AppDashboard import AppDashboard
# import next.database_client.DatabaseAPIHTTP as db
# import next.logging_client.LoggerHTTP as ell

class MyAppDashboard(AppDashboard):

    def __init__(self,db,ell):
        AppDashboard.__init__(self, db, ell)
