import gymnasium
import random
import statistics
import jax
import math
import numpy as np

def policy(theta, state):
    sigma = math.exp(theta[0])
    mu = theta[1] + state @ theta[2:]
    return statistics.NormalDist(mu, sigma).samples(1)[0]

def policy_mean(theta, state):
    mu = theta[1] + state @ theta[2:]
    return mu.item()

def logpi(theta, action, state):
    sigma = jax.numpy.exp(theta[0])
    mu = theta[1] + state @ theta[2:]
    return jax.scipy.stats.norm.logpdf(action, mu, sigma)

def episod():
    Return = 0
    state, info = env.reset()
    while True:
        action = policy_mean(theta, state)
        state, reward, terminated, truncated, info = env.step(action > 0)
        Return += reward
        if terminated or truncated:
            break
    return Return

seed = gymnasium.utils.seeding.np_random()
dlogpi_dtheta = jax.grad(logpi)
env = gymnasium.make("CartPole-v1")
random.seed(12345)
gamma = 0.99
n_states, = env.observation_space.shape
theta = jax.numpy.zeros(n_states + 2, dtype=float)
alpha = 0.001
n_episod = 0
while True:
    S = {}
    A = {}
    R = {}
    t = 0
    state, info = env.reset(seed=12345)
    while True:
        S[t] = state
        A[t] = action = policy(theta, S[t])
        state, reward, terminated, truncated, info = env.step(action > 0)
        t += 1
        R[t] = reward
        if terminated or truncated:
            break
    T = t
    if n_episod > 0 and n_episod % 100 == 0:
        Return = statistics.fmean(episod() for i in range(100))
        print("% 8d %8.2f %s" % (n_episod, Return, theta))
    for t in range(T):
        G = statistics.fsum(gamma**(k - t - 1) * R[k]
                            for k in range(t + 1, T + 1))
        dlog = dlogpi_dtheta(theta, A[t], S[t])
        theta += alpha * gamma**t * G * dlog
    n_episod += 1
