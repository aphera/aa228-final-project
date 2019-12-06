import copy
import itertools
import random
from timeit import default_timer as timer
import jsonpickle
import numpy as np

from mido import MidiFile

from kalman_filter import KalmanFilterParameters, calculate_beat, State, calculate_error, ResultMetrics, plot_results
from midi_reader import get_observations, get_observations_for_files_in_directory


def define_possible_q(k_f_p, increment):
    new_k_f_p = copy.deepcopy(k_f_p)
    new_k_f_p.q_per_second += increment
    return new_k_f_p


def define_possible_observation_error_weight(k_f_p, increment):
    new_k_f_p = copy.deepcopy(k_f_p)
    new_k_f_p.observation_error_weight += increment
    return new_k_f_p


def define_possible_observation_weight_vector(k_f_p, increment):
    possible_parameters = []
    for i in range(0, k_f_p.observation_weight_vector.shape[0]):
        new_k_f_p = copy.deepcopy(k_f_p)
        new_k_f_p.observation_weight_vector[i] += increment
        possible_parameters.append(new_k_f_p)
    return possible_parameters


def define_possible_parameters(k_f_p, increment):
    possible_parameters = []
    possible_parameters.append(define_possible_q(k_f_p, increment))
    possible_parameters.append(define_possible_observation_error_weight(k_f_p, increment))
    possible_parameters = possible_parameters + define_possible_observation_weight_vector(k_f_p, increment)
    return possible_parameters


def sample(k_f_p, observations):
    try:
        result_metrics = ResultMetrics()
        calculate_beat(State(), k_f_p, observations, result_metrics)
        total_error = calculate_error(result_metrics)
        return 1 / total_error
    except np.linalg.LinAlgError:
        return float("-inf")


def local_search(k_f_p, observations_list, check_observations, increment, max_iterations):
    start = timer()
    previous_reward = float("-inf")
    reward = sample(k_f_p, itertools.chain.from_iterable(observations_list))
    i = 0
    steps_without_improvements = 0
    max_steps_without_improvements = 4
    while (reward > previous_reward or steps_without_improvements < max_steps_without_improvements) and i < max_iterations:
        i += 1
        previous_reward = reward
        possible_parameters = define_possible_parameters(k_f_p, increment) + define_possible_parameters(k_f_p,
                                                                                                        -increment)
        random.shuffle(observations_list)
        for new_k_f_p in possible_parameters:
            new_reward = sample(new_k_f_p, list(itertools.chain.from_iterable(observations_list)))
            if new_reward > reward:
                reward = new_reward
                k_f_p = new_k_f_p
                steps_without_improvements = 0
        if reward == previous_reward:
            steps_without_improvements += 1
            print(f"No improvment for {steps_without_improvements} steps")
        print(f"error:\n{1 / reward}")
        print(f"check error:\n{1 / sample(k_f_p, check_observations)}")
        print(f"q_per_second:\n{k_f_p.q_per_second}")
        print(f"observation_weight_vector:\n{k_f_p.observation_weight_vector}")
        print(f"observation_error_weight:\n{k_f_p.observation_error_weight}")
        print(f"iteration: {i}")
        print()
        write_parameters(k_f_p)
    end = timer()
    print(f"Time took to local search {str(end - start)}")
    return k_f_p


FILE_NAME = "k_f_p.txt"


def write_parameters(parameters):
    with open(FILE_NAME, "a") as f:
        f.write("{}\n".format(jsonpickle.encode(parameters)))


def read_parameters_or_create_new():
    file = open(FILE_NAME, "r")
    lines = file.readlines()
    if lines:
        return jsonpickle.decode(lines[-1])
    return KalmanFilterParameters()


def coordinate_local_search():
    k_f_p = read_parameters_or_create_new()
    print(k_f_p.observation_weight_vector)
    observations = get_observations_for_files_in_directory("Wtc2midi")
    check_observations = get_observations(MidiFile("bwv988.mid"))

    print(f"Starting error:\n{1 / sample(k_f_p, itertools.chain.from_iterable(observations))}")
    print(f"Starting check error:\n{1 / sample(k_f_p, check_observations)}")

    best_k_f_p = local_search(k_f_p, observations, check_observations, 0.3, 15)

    result_metrics = ResultMetrics()
    calculate_beat(State(), best_k_f_p, itertools.chain.from_iterable(observations), result_metrics)
    plot_results(result_metrics)


coordinate_local_search()
