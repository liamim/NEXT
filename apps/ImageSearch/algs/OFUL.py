from __future__ import division
import numpy as np
import next.utils as utils
import time

# TODO: change this to 1
reward_coeff = 1.00

def timeit(fn_name=''):
    def timeit_(func, *args, **kwargs):
        def timing(*args, **kwargs):
            start = time.time()
            r = func(*args, **kwargs)
            utils.debug_print('')
            utils.debug_print("function {} took {} seconds".format(fn_name, time.time() - start))
            utils.debug_print('')
            return r
        return timing
    return timeit_


def CalcSqrtBeta(d, t, scale, R, ridge, delta, S_hat=1.0):
    return scale * (R * np.sqrt(d * np.log((1 + t / (ridge * d)) / delta)) + np.sqrt(ridge) * S_hat)


class MyAlg:
    app_id = 'ImageSearch'
    def initExp(self, butler, params=None, n=None, R=None, ridge=None,
                failure_probability=None):
        """
        initialize the experiment

        (int) n : number of arms
        (float) R : sub-Gaussian parameter, e.g. E[exp(t*X)]<=exp(t^2 R^2/2),
                    defaults to R=0.5 (satisfies X \in [0,1])
        (float) failure_probability : confidence
                imp note: delta
        (dict) params : algorithm-specific parameters (if none provided in
                        alg_list of init experiment, params=None)

        Expected output (comma separated):
          (boolean) didSucceed : did everything execute correctly
        """
        # setting the target matrix, a description of each target
        # X = np.asarray(params['X'])
        #X = get_feature_vectors()
        X = butler.db.X
        # theta_star = np.asarray(params['theta_star'])
        d = X.shape[1]  # number of dimensions in feature
        n = X.shape[0]

        #lambda_ = ridge
        lambda_ = 1.0
        R = 1.0

        # initial sampling arm
        # theta_hat = X[:, np.random.randint(X.shape[1])]
        # theta_hat = np.random.randn(d)
        # theta_hat /= np.linalg.norm(theta_hat)

        to_save = {'R': R, 'd': d, 'n': n,
                   'lambda_': lambda_,
                   'total_pulls': 0.0,
                   'rewards': [],
                   'ask_indices': range(n),
                   'arms_pulled': [],
                   'failure_probability': failure_probability}

        for name in to_save:
            butler.algorithms.set(key=name, value=to_save[name])

        return True

    @timeit(fn_name='alg:getQuery')
    def getQuery(self, butler, participant_uid):
        """
        A request to ask which index/arm to pull

        Expected input:
          (list of int) do_not_ask_list : indices in {0,...,n-1} that the
                algorithm must not return. If there does not exist an index
                that is not in do_not_ask_list then any index is acceptable
                (this changes for each participant so they are not asked the
                same question twice)

        Expected output (comma separated):
          (int) target_index : idnex of arm to pull (in 0,n-1)

         particpant_doc is butler.participants corresponding to this
         participant

        if we want, we can find some way to have different arms
        pulled using the butler
        """
        expected_rewards = np.asarray(butler.participants.get(uid=participant_uid, key='expected_rewards'))
        do_not_ask = butler.participants.get(uid=participant_uid, key='do_not_ask')
        utils.debug_print('dna: ', do_not_ask)
        expected_rewards[np.asarray(do_not_ask)] = -np.inf
        i_x = np.argmax(expected_rewards)
        butler.participants.append(uid=participant_uid,
                                   key='do_not_ask', value=i_x)
        return i_x

    @timeit(fn_name='alg:processAnswer')
    def processAnswer(self, butler, target_id=None,
                      target_reward=None, participant_uid=None):
        """
        reporting back the reward of pulling the arm suggested by getQuery

        Expected input:
          (int) target_index : index of arm pulled
          (int) target_reward : reward of arm pulled

        Expected output (comma separated):
          (boolean) didSucceed : did everything execute correctly
        """

        if not target_id:
            participant_doc = butler.participants.get(uid=participant_uid)
            # utils.debug_print('pargs in processAnswer:', participant_doc)
            # X = get_feature_vectors()
            X = butler.db.X
            participant_uid = participant_doc['participant_uid']

            n = X.shape[0]
            d = X.shape[1]
            lambda_ = butler.algorithms.get(key='lambda_')

            utils.debug_print('setting t for first time')
            target_id = butler.participants.get(uid=participant_uid, key='i_hat')
            expected_rewards = X.dot(X[target_id,:])
            expected_rewards[target_id] = -np.inf
            data = {'t': 1,
                    'b': np.zeros(d),
                    'invV': np.eye(d)/lambda_,
                    'x_invVt_norm': np.ones(n)/lambda_,
                    'do_not_ask': [target_id],
                    'expected_rewards': expected_rewards
                    }
            participant_doc.update(data)
            #for key in data.keys():
            #    butler.participants.set(uid=participant_uid, key=key)

            butler.participants.set_many(uid=participant_doc['participant_uid'],
                                         key_value_dict=participant_doc)

            return True

        task_args = {
            'target_id': target_id,
            'target_reward': target_reward,
            'participant_uid': participant_uid
        }


        butler.job('modelUpdate', task_args, ignore_result=True)

        return True

    def modelUpdate(self, butler, task_args):
        target_id = task_args['target_id']
        target_reward = task_args['target_reward']
        participant_uid = task_args['participant_uid']

        participant_doc = butler.participants.get(uid=participant_uid)
        X = butler.db.X
        reward = target_reward
        participant_uid = participant_doc['participant_uid']
        i_hat = butler.participants.get(uid=participant_uid, key='i_hat')

        d = X.shape[1]
        lambda_ = butler.algorithms.get(key='lambda_')
        R = butler.algorithms.get(key='R')

        butler.participants.increment(uid=participant_uid, key='t')

        scale = 1.0

        # this makes sure the reward propogates from getQuery to processAnswer
        b = np.array(participant_doc['b'], dtype=float)
        do_not_ask = participant_doc['do_not_ask']
        invV = np.array(participant_doc['invV'], dtype=float)
        x_invVt_norm = np.array(participant_doc['x_invVt_norm'], dtype=float)

        arm_pulled = X[target_id, :]
        utils.debug_print('size of X:', X.shape)
        utils.debug_print('size of arm_pulled: ', arm_pulled.shape)

        u = invV.dot(arm_pulled)
        utils.debug_print('size of np.dot(X, u):', np.dot(X, u).shape)
        invV -= np.outer(u, u) / (1 + np.inner(arm_pulled, u))

        x_invVt_norm -= np.dot(X, u) ** 2 / (1 + np.inner(arm_pulled, u))

        b += reward * arm_pulled
        theta_hat = X[i_hat, :] + invV.dot(b)

        sqrt_beta = CalcSqrtBeta(d, participant_doc['t'], scale, R, lambda_,
                                 butler.algorithms.get(key='failure_probability'))
        expected_rewards = np.dot(X, theta_hat) + sqrt_beta * np.sqrt(x_invVt_norm)
        expected_rewards[do_not_ask] = -np.inf

        # save the results
        data = {'x_invVt_norm': x_invVt_norm,
                'b': b,
                'invV': invV,
                'theta_hat': theta_hat,
                'expected_rewards': expected_rewards
                }
        participant_doc.update(data)

        butler.participants.set_many(uid=participant_doc['participant_uid'],
                                     key_value_dict=participant_doc)
        return True

    def getModel(self, butler):
        """
        uses current model to return empirical estimates with uncertainties

        Expected output:
          (list float) mu : list of floats representing the emprirical means
          (list float) prec : list of floats representing the precision values
                              (or standard deviation)
        """
        # TODO: I can't see the results without this
        # (and we also need to change the label name if we want to see results,
        # correct?)
        return 0.5  # mu.tolist(), prec


