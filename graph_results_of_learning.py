import matplotlib.pyplot as plt
import itertools

import jsonpickle
from mido import MidiFile

from kalman_filter import ResultMetrics, calculate_beat, State, calculate_error
from midi_reader import get_observations_for_files_in_directory, get_observations


def read_parameters():
    file = open("parameter_updates.txt", "r")
    lines = file.readlines()
    parameters = []
    for line in lines:
        parameters.append(jsonpickle.decode(line))
    return parameters


def graph_results():
    all_wtc = get_observations_for_files_in_directory("Wtc2midi")
    wtc_observations = list(itertools.chain.from_iterable(all_wtc))

    goldberg_observations = get_observations(MidiFile("bwv988.mid"))

    parameters = read_parameters()
    wtc_errors = []
    goldberg_errors = []
    for parameter in parameters:
        wtc_result_metrics = ResultMetrics()
        calculate_beat(State(), parameter, wtc_observations, wtc_result_metrics)
        wtc_errors.append(calculate_error(wtc_result_metrics))

        goldberg_result_metrics = ResultMetrics()
        calculate_beat(State(), parameter, goldberg_observations, goldberg_result_metrics)
        goldberg_errors.append(calculate_error(goldberg_result_metrics))

    plt.figure()
    plt.plot(wtc_errors, 'b-')
    plt.plot(goldberg_errors, 'r-')
    plt.show()

