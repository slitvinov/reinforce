import argparse
import os.path
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


class ShortCorridor:
    start_state = 0
    goal_state = 3
    num_states = 4
    num_actions = 2
    left = 0
    right = 1

    @staticmethod
    def init():
        return ShortCorridor.start_state

    @staticmethod
    def reset():
        return ShortCorridor.start_state

    @staticmethod
    def step(state, action):
        assert ShortCorridor.start_state <= state < ShortCorridor.goal_state
        assert action == ShortCorridor.left or action == ShortCorridor.right

        if action == ShortCorridor.left:
            if state == 1:
                state += 1
            elif ShortCorridor.start_state < state:
                state -= 1
        elif action == ShortCorridor.right:
            if state == 1:
                state -= 1
            elif state < ShortCorridor.goal_state:
                state += 1
        else:
            raise ValueError('Invalid Action!')

        if state == ShortCorridor.goal_state:
            return -1, None
        else:
            return -1, state


def pi(x_s):
    preferences = theta.dot(x_s)
    preferences = preferences - preferences.max()
    exp_prefs = np.exp(preferences)
    return exp_prefs / np.sum(exp_prefs)


def select_action(x_s):
    return np.random.choice(2, p=pi(x_s).squeeze())


def eligibility_vector(a, s):
    return x(
        s,
        a) - pi(x(s)) * (x(s, ShortCorridor.left) + x(s, ShortCorridor.right))


def x(s, a=None):
    if a is None:
        return np.array([[1]])
    elif a == ShortCorridor.right:
        return np.array([[0], [1]])
    elif a == ShortCorridor.left:
        return np.array([[1], [0]])
    else:
        raise ValueError('Invalid Action!')


alpha = 2**-13
max_timesteps = 1000
theta = np.log([[19], [1]])
num_episodes = 10000
returns = []
for episode_num in range(num_episodes):
    episode = []
    g = 0.0
    t = 0
    s = ShortCorridor.init()
    x_s = x(s)
    while s is not None and t < max_timesteps:
        a = select_action(x_s)
        r_prime, s_prime = ShortCorridor.step(s, a)
        episode.append((s, a, r_prime))
        s = s_prime
        g = g + r_prime
        t = t + 1
    returns.append(g)
    gt = g
    for t in range(len(episode)):
        s, a, r_prime = episode[t]
        x_s = x(s)
        ev = eligibility_vector(a, s)
        theta += alpha * g * eligibility_vector(a, s)
        gt = gt - r_prime
print(returns[-10:])
