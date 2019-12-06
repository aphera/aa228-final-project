import copy
from timeit import default_timer as timer
import jsonpickle

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
    result_metrics = ResultMetrics()
    calculate_beat(State(), k_f_p, observations, result_metrics)
    total_error = calculate_error(result_metrics)
    return 1 / total_error


def local_search(k_f_p, observations, check_observations, increment, max):
    start = timer()
    previous_reward = float("-inf")
    reward = sample(k_f_p, observations)
    i = 0
    while reward > previous_reward and i < max:
        i += 1
        previous_reward = reward
        possible_parameters = define_possible_parameters(k_f_p, increment) + define_possible_parameters(k_f_p, -increment)
        # possible_parameters = define_possible_parameters(k_f_p, increment)
        for new_k_f_p in possible_parameters:
            new_reward = sample(new_k_f_p, observations)
            if new_reward > reward:
                reward = new_reward
                k_f_p = new_k_f_p
        print(1 / reward)
        print(1 / sample(k_f_p, check_observations))
        print(k_f_p.observation_error_weight)
        print()
    end = timer()
    print(f"Time took to local search {str(end - start)}")
    return reward, k_f_p


def coordinate_local_search():
    # midi_file =
    # midi_file = MidiFile("988-v25.mid")
    # midi_file = MidiFile("cs1-1pre.mid")
    # midi_file = MidiFile("vs1-1ada.mid")
    # observations = get_observations(MidiFile("bwv988.mid"))
    observations = get_observations_for_files_in_directory("Wtc2midi")
    check_observations = get_observations(MidiFile("vs1-1ada.mid"))
    file = open("parameters_1.txt", "r")
    k_f_p = jsonpickle.decode(file.read())
    print(f"Starting error:\n{1 / sample(k_f_p, observations)}")
    print(f"Starting check error:\n{1 / sample(k_f_p, check_observations)}")

    ok_reward, ok_k_f_p = local_search(k_f_p, observations, check_observations, 1.0, 5)
    print(f"Ok error:\n{1 / ok_reward}")
    better_reward, better_k_f_p = local_search(ok_k_f_p, observations, check_observations, 10.0, 5)
    print(f"Better error:\n{1 / better_reward}")
    best_reward, best_k_f_p = local_search(better_k_f_p, observations, check_observations, 1.0, 5)
    print(f"Best error:\n{1 / best_reward}")
    print(best_k_f_p)
    return best_k_f_p


def write_parameters(parameters, filename):
    with open(filename, "w") as f:
            f.write("{}\n".format(parameters))


b_k_f_p = coordinate_local_search()
print(b_k_f_p)

os = get_observations(MidiFile("bwv988.mid"))
so_good = sample(b_k_f_p, os)
print(1 / so_good)

rm = ResultMetrics()
calculate_beat(State(), b_k_f_p, os, rm)

write_parameters(jsonpickle.encode(b_k_f_p), "parameters_2.txt")
plot_results(rm)
