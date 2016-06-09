"""
RoundRobin app implements CardinalBanditsPureExplorationPrototype
author: Kevin Jamieson
last updated: 12/15/2015
"""

import numpy
import numpy.random
from next.apps.Apps.CardinalBanditsPureExploration.Prototype import CardinalBanditsPureExplorationPrototype

class RoundRobin(CardinalBanditsPureExplorationPrototype):

  def initExp(self,butler,n,R,failure_probability,params):
    butler.algorithms.set(key='n', value=n)
    butler.algorithms.set(key='failure_probability',value=failure_probability)
    butler.algorithms.set(key='R',value=R)
    arm_key_value_dict = {}
    for i in range(n):
      arm_key_value_dict['Xsum_'+str(i)] = 0.
      arm_key_value_dict['X2sum_'+str(i)] = 0.
      arm_key_value_dict['T_'+str(i)] = 0.
    arm_key_value_dict.update({'total_pulls':0,'generated_queries_cnt':-1})
    butler.algorithms.increment_many(key_value_dict=arm_key_value_dict)

    return True

  
  def getQuery(self,butler,participant_dict,**kwargs):
    do_not_ask_hash = {key: True for key in participant_dict.get('do_not_ask_list',[])}
    
    # n = butler.algorithms.get(key='n')
    # cnt = butler.algorithms.increment(key='generated_queries_cnt',value=1)
    # The following line performs the previous two lines in one query to the database
    kv_dict = butler.algorithms.increment_many(key_value_dict={'n':0,'generated_queries_cnt':1})
    n = kv_dict['n']
    cnt = kv_dict['generated_queries_cnt']

    k=0
    while k<n and do_not_ask_hash.get(((cnt+k)%n),False):
      k+=1
    if k<n:
      index = (cnt+k)%n
    else:
      index = numpy.random.choice(n)

    return index

  def processAnswer(self,butler,target_id,target_reward): 
    butler.algorithms.increment_many(key_value_dict={'Xsum_'+str(target_id):target_reward,'X2sum_'+str(target_id):target_reward*target_reward,'T_'+str(target_id):1,'total_pulls':1})
    
    return True

  def getModel(self,butler):
    key_value_dict = butler.algorithms.get()
    R = key_value_dict['R']
    n = key_value_dict['n']
    sumX = [key_value_dict['Xsum_'+str(i)] for i in range(n)]
    sumX2 = [key_value_dict['X2sum_'+str(i)] for i in range(n)]
    T = [key_value_dict['T_'+str(i)] for i in range(n)]

    mu = numpy.zeros(n)
    prec = numpy.zeros(n)
    for i in range(n):
      if T[i]==0 or mu[i]==float('inf'):
        mu[i] = -1
        prec[i] = -1
      elif T[i]==1:
        mu[i] = float(sumX[i]) / T[i]
        prec[i] = R
      else:
        mu[i] = float(sumX[i]) / T[i]
        prec[i] = numpy.sqrt( float( max(1.,sumX2[i] - T[i]*mu[i]*mu[i]) ) / ( T[i] - 1. ) / T[i] )
    
    return mu.tolist(),prec.tolist()
