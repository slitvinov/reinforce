import gymnasium
import matplotlib.pyplot as plt
import random
import statistics
import math
import numpy as np

class Env:
    def __init__(self):
        self.E = 1.0
        self.m = 1.0
        self.k = 1.0
        self.dt = 0.01
        self.reset()

    def reset(self):
        self.x = 5
        self.u = 0
        self.t = 0
        return self.state(), None

    def step(self, action):
        dt = self.dt
        m = self.m
        f = action
        self.u += dt * (f - self.k * self.x) / m
        self.x += self.u * dt
        self.t += dt

        state = self.state()
        reward = self.reward()
        done = self.done()
        info = None
        return state, reward, done, done, info

    def state(self):
        return np.array([self.x, self.u])

    def reward(self):
        return 1.0 / (1e-1 + abs(0.5 * self.m * self.u**2 - self.E))

    def done(self):
        return self.t >= 2 * np.sqrt(self.k/self.m)

def policy(th, state):
    mu = th[2] * state + th[1]
    sigma = th[0]
    return statistics.NormalDist(mu, sigma).samples(1)[0]


def dlogpi_dth(th, action, state):
    mu = th[2] * state + th[1]
    sigma = th[0]
    return \
        -((sigma-mu+action)*(sigma+mu-action))/sigma**3, \
        -(mu-action)/sigma**2, \
        -(mu-action)*state/sigma**2


#env = gymnasium.make("CartPole-v1")
env = Env()
th = [0.1, 0.0, 0.0]
alpha = 1e-6
n_episod = 0
while True:
    S = {}
    A = {}
    R = {}
    t = 0
    state, info = env.reset()
    while True:
        S[t] = state[0]
        A[t] = action = policy(th, S[t])
        state, reward, terminated, truncated, info = env.step(action)
        t += 1
        R[t] = reward
        if terminated or truncated:
            break
    T = t
    if n_episod > 0 and n_episod % 100 == 0:
        Return = math.fsum(R[k] for k in range(1, T + 1))
        print("% 8d %8.2f [%s]" % (n_episod, Return, th))
    for t in range(T):
        G = math.fsum(R[k] for k in range(t + 1, T + 1))
        dlog = dlogpi_dth(th, A[t], S[t])
        s2 = th[0] * th[0]
        th[0] += alpha * s2 * G * dlog[0]
        th[1] += alpha * s2 * G * dlog[1]
        th[2] += alpha * s2 * G * dlog[2]
    n_episod += 1
