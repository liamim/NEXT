from __future__ import division, print_function
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from scipy.io import loadmat
import scipy.stats as stats
from scipy.optimize import curve_fit
# plt.style.use('seaborn-poster')
import code
import time, sys
norm = np.linalg.norm

# TODO: update to higher dimensions, see if follows sqrt(t) curve

TAKE_SIGN = False
USE_DO_NOT_ASK = True

def argmax_reward(X, theta, invV, beta, k=1, do_not_ask=None):
    r"""
    Loop over all columns of X to solve this equation:

        \widehat{x} = \arg \min_{x \in X} x^T theta + k x^T V^{-1} x
    """
    inv = np.linalg.inv
    norm = np.linalg.norm
    sqrt = np.sqrt

    start = time.time()
    rewards = X.T.dot(theta) + sqrt(k)*sqrt(beta)
    rewards = np.asarray(rewards)

    if USE_DO_NOT_ASK:
        rewards[do_not_ask] = -np.inf
    return X[:, np.argmax(rewards)], np.argmax(rewards)

def choose_answer(i_x, labels=None):
    if labels[i_x]:
        return 1
    return -1

def OFUL(X=None, R=None, theta_hat=None, theta_star=None, invV=None, S=1, T=25,
         d=None, n=None, lambda_=None, PRINT=False, labels=None):
    """"
    X : x
    R: x
    theta_hat : algorithms, key='theta_hat'
    theta_star : a param the user passes in
    V : 
    S : unused
    T : num_tries
    d : x
    n : x
    lambda_ : can be hard coded for now
    """
    norm = np.linalg.norm
    if PRINT:
        print("theta_star = {}".format(theta_star))

    # On NEXT, only save one reward and one arm
    rewards, arms = [], []
    b = np.zeros(d)
    beta = np.ones(n) / lambda_
    if PRINT:
        print("Arms = \n{}".format(X.T))
    V = lambda_ * np.eye(d)
    rel_errors = []
    for t in 1 + np.arange(T):
        k = R * np.sqrt(d*np.log((1 + t/lambda_) / delta)) + np.sqrt(lambda_)

        x, i_x = argmax_reward(X, theta_hat, invV, beta, k=k,
                               do_not_ask=arms)
        # TODO: his R needs to be tuned!
        rewards += [choose_answer(i_x, labels=labels)]
        arms += [i_x]
        print("arm pulled = {} @ iteration {}".format(i_x, t))

        u = invV.dot(x)
        invV -= np.outer(u, u) / (1 + np.inner(x, u))
        beta -= (X.T.dot(u))**2 / (1 + beta[i_x])

        b += rewards[-1] * x
        theta_hat = invV.dot(b)


        if PRINT:
            print("||theta_hat - theta_star|| = {} @ {}".format(norm(theta_hat - theta_star), t))
        # rel_errors += [norm(theta_hat)]
        rel_errors += [norm(theta_hat - theta_star)]
    return theta_hat, np.asarray(rewards), arms, rel_errors

# all for (n, d) = (2, 500)
# np.random.seed(1)  # quick convergence
# np.random.seed(5)
# np.random.seed(42)  # convergence to low reward
# np.random.seed(43)  # again quick convergence
np.random.seed(42)
T = 50
d, n = (int(5e0), int(1e4))
R = 2  # the std.dev for a sub-gaussian random variable
lambda_ = 1 / d

delta = 0.1  # failure probability

# The arms to pull. The columns are arms, the rows are features
X = np.random.randn(d, n)
X = normalize(X, axis=0)

X = np.load('/Users/scott/Dropbox/Public/features_allshoes_8_normalized.npy')
d, n = X.shape
print(X.shape)

i_star = 30274;#4051;#moderately hard red boot#2960;# hard red boot #2227; # that red rainboot
i_star = 2228
i_star -= 1

# This file contains labels, such as "red shoe"
labels = loadmat('/Users/scott/Dropbox/Public/Labels.mat')['Labels']
redLabels = loadmat('/Users/scott/Dropbox/Public/ColorLabel_new.mat')

boots = labels[:, 0] == labels[i_star, 0]
red = redLabels['RedLabel'].flat[:]
red = red == red[i_star]
answers = np.logical_and(boots, red)

# theta can either be zeros or random
# initial approximation
# theta_star = X[:, i_star].copy()
x_star = X[:, i_star].copy()

theta_hat = np.random.randn(d)
theta_hat /= np.linalg.norm(theta_hat)
theta_hat = np.zeros(d)

invV = np.eye(d) / lambda_
theta_star = invV.dot(x_star)
theta_star = X[:, i_star].copy()
r_star = (X.T.dot(theta_star)).max()
r_star = 1

theta_hat, rewards, arms, rel_errors = OFUL(X=X, R=R, theta_hat=theta_hat,
                                theta_star=theta_star, invV=invV, d=d, n=n, T=T,
                                lambda_=lambda_, labels=answers)
print("||theta_final - theta_star|| = {}".format(norm(theta_hat -
    theta_star)))
plt.figure(figsize=(16, 8))
plt.subplot(1, 3, 1)
plt.title('Rewards')
plt.plot(np.cumsum(r_star - rewards))
plt.ylabel('cumsum(reward_star - rewards)')
plt.xticks(rotation=90)

plt.subplot(1, 3, 2)
plt.title('Relative error between\n truth $\\theta^\\star$ and estimate $\\widehat{\\theta}$')
plt.ylabel(r'$\|\theta^\star - \widehat{\theta} \|_2$')
plt.plot(rel_errors)
plt.xticks(rotation=90)

plt.subplot(1, 3, 3)
plt.plot(rewards, 'o')
print(sum(rewards == 1))
print(sum(rewards == -1))
plt.title('Answers at each iteration')
plt.savefig('take.sign={}_use.do.not.ask={}.png'.format(TAKE_SIGN,
                                                        USE_DO_NOT_ASK))
plt.show()
