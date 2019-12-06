import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from src.midi_reader import get_observations


class ResultMetrics:
    def __init__(self):
        # tempo at each step
        self.xs = []
        # observation at each step
        self.zs = []
        # index of observation array chosen during observation
        self.ss = []
        # variance at each step
        self.ps = []
        # if known, actual tempo at each step
        self.ts = []


def get_z_and_r(observation, b_p, result_metrics=None):
    if observation.seconds_since_last_beat == 0:
        return None
    observed_bpm = 60 / observation.seconds_since_last_beat
    scaled_observation_vector = b_p.observation_scalar_vector * observed_bpm
    current_bpm_estimate = b_p.h.dot(b_p.x)[0, 0]
    idx = (np.abs(scaled_observation_vector - current_bpm_estimate)).argmin()
    if result_metrics:
        result_metrics.ss.append(idx)
    z = scaled_observation_vector[idx]
    r = b_p.observation_weight_vector[idx] + b_p.observation_error_weight * np.power(current_bpm_estimate - z, 2)
    return z, r


class BeatParameters:
    def __init__(self,
                 x=np.array([[80.0]]),
                 h=np.array([[1.0]]),
                 p=np.array([[160.0]]),
                 q_per_second=16.0,
                 z=np.array([[0.0]]),
                 r=np.array([[0.0]]),
                 observation_scalar_vector=np.arange(0.125, 4.25, 0.125),
                 observation_error_weight=1.0
                 ):
        # bpm
        self.x = x
        self.h = h
        self.h_t = h.transpose()
        # our covariance matrix
        self.p = p
        # Q grows the covariance over time
        self.q_per_second = q_per_second
        # observation
        self.z = z
        # observation covariance
        self.r = r
        self.observation_scalar_vector = observation_scalar_vector
        self.observation_weight_vector = np.ones(observation_scalar_vector.shape)
        self.observation_error_weight = observation_error_weight


def calculate_beat(b_p, observations, result_metrics=None):
    start = timer()
    for observation in observations:
        b_p.p = b_p.p + b_p.q_per_second * observation.seconds_since_last_beat
        z_and_r = get_z_and_r(observation, b_p, result_metrics)
        if z_and_r is not None:
            b_p.z[0, 0] = z_and_r[0]
            b_p.r[0, 0] = z_and_r[1]

            k_prime = b_p.p.dot(b_p.h_t).dot(np.linalg.inv(b_p.h.dot(b_p.p).dot(b_p.h_t) + b_p.r))
            b_p.x = b_p.x + k_prime.dot(b_p.z - b_p.h.dot(b_p.x))
            b_p.p = b_p.p - k_prime.dot(b_p.h.dot(b_p.p))
        if result_metrics:
            result_metrics.xs.append(b_p.x[0, 0])
            result_metrics.zs.append(b_p.z[0, 0])
            result_metrics.ps.append(b_p.p[0, 0])
            result_metrics.ts.append(observation.actual_bpm)

    end = timer()
    print(f"Time took kalman filter {str(end - start)}")


error_vector = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])


def calculate_error(result_metrics):
    start = timer()
    error = 0
    samples = len(result_metrics.xs)
    for i, x in enumerate(result_metrics.xs):
        actual_bpm = result_metrics.ts[i]
        # Overloading the meaning of this function but it gives us what we want
        scaled_error_vector = error_vector * x
        idx = (np.abs(scaled_error_vector - actual_bpm)).argmin()
        adjusted_estimate = scaled_error_vector[idx]
        error += np.power(actual_bpm - adjusted_estimate, 2) / samples
    end = timer()
    print(f"Time took calculate error {str(end - start)}")
    return error


os = get_observations()
rm = ResultMetrics()
calculate_beat(BeatParameters(), os, rm)
total_error = calculate_error(rm)
print(f"Error:\n{total_error}")


def plot_results(result_metrics):
    plt.figure()
    # tempo at each step
    plt.plot(result_metrics.xs, 'b-')
    # observation at each step
    # plt.plot(result_metrics.zs, 'r-')
    # if known, actual tempo at each step
    plt.plot(result_metrics.ts, 'g-')

    # unique, counts = np.unique(np.array(result_metrics.ss), return_counts=True)
    # plt.plot(unique, counts)

    # plt.figure()
    # variance at each step
    # plt.plot(ps[-4000:],'y-')

    plt.show()


plot_results(rm)
