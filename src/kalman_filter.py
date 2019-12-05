import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from src.midi_reader import get_observations

# tempo at each step
xs = []
# pre slide matched observation at each step
ss = []
# observation at each step
zs = []
# variance at each step
ps = []
# if known, actual tempo at each step
ts = []


observation_base_vector_tuples = np.arange(0.25, 4.25, 0.25)


def find_z_in_vector_and_compute_r(observation_vector, bpm_estimate):
    idx = (np.abs(observation_vector - bpm_estimate)).argmin()
    z_bpm = observation_vector[idx]
    diff = abs(bpm_estimate - z_bpm)
    return np.array([[z_bpm]]), np.array([[1.0 + diff * diff]])


def get_z_and_r(observation, bpm_estimate):
    if observation.seconds_since_last_beat == 0:
        return None
    observed_bpm = 60 / observation.seconds_since_last_beat
    ss.append(observed_bpm)
    ts.append(observation.actual_bpm)
    return find_z_in_vector_and_compute_r(observation_base_vector_tuples * observed_bpm, bpm_estimate)


def calculate_beat():
    # bpm
    x = np.array([[80.0]])
    h = np.array([[1.0]])
    h_t = h.transpose()

    # our covariance matrix
    p = np.array([[160.0]])

    # Q grows the covariance over time
    q_per_second = 16.0

    observations = get_observations()

    start = timer()
    for observation in observations:
        p = p + q_per_second * observation.seconds_since_last_beat
        z_and_r = get_z_and_r(observation, h.dot(x)[0, 0])
        if z_and_r is not None:
            z, r = z_and_r

            k_prime = p.dot(h_t).dot(np.linalg.inv(h.dot(p).dot(h_t) + r))
            x = x + k_prime.dot(z - h.dot(x))
            p = p - k_prime.dot(h.dot(p))
            zs.append(z[0, 0])
            xs.append(x[0, 0])

        ps.append(p[0, 0])

    end = timer()
    print(f"Time took kalman filter {str(end - start)}")
    print(x[0, 0])
    print(p[0, 0])


calculate_beat()

plt.figure()
# tempo at each step
plt.plot(xs, 'b-')
# observation at each step
# plt.plot(zs, 'r-')
# if known, actual tempo at each step
plt.plot(ts, 'g-')

# plt.figure()
# variance at each step
# plt.plot(ps[-4000:],'y-')

plt.show()
