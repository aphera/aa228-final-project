import numpy as np
from src.kalman_filter import KalmanFilterParameters, calculate_beat, State, calculate_error, ResultMetrics
from src.midi_reader import get_observations

T = set()
N = {}
Q = {}
q_action_space = [-1, 1]
q_policy = None
c = 0.01
os = get_observations()


def q_update_state(state, action):
    state.q += action
    return state



def sample(update_state, state, action):
    new_state = update_state(state, action)
    rm = ResultMetrics()
    calculate_beat(State(), new_state, os, rm)
    total_error = calculate_error(rm)
    return 1 / total_error


def select_action(state, depth, loops):
    for _ in range(0, loops):
        q = simulate(state, depth, q_action_space, q_policy)
        print(q)


def find_best_action(state, action_space):
    best_reward = float("-inf")
    best_action = 0
    for action in action_space:
        immediate_reward = Q[state][action]
        exploration_bonus = c * np.sqrt(np.log(sum(N[state].values())) / N[state][action])
        reward = immediate_reward + exploration_bonus
        if best_reward < reward:
            best_reward = reward
            best_action = action
    return best_action



def simulate(state, depth, action_space, policy, update_state):
    if depth == 0:
        return 0
    if state not in T:
        N[state] = {}
        Q[state] = {}
        for action in action_space:
            N[state][action] = 0
            Q[state][action] = 0
        T.add(state)
        return rollout(state, depth, policy)
    action = find_best_action(state, action_space)
    new_state, reward = sample(update_state, state, action)


def rollout(state, depth, policy):
    pass


select_action(KalmanFilterParameters(), 10, 10)
