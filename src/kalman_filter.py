import random

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# samples_per_second = 44100
samples_per_second = 200

# bpm
x = np.array([[120.0]])
h = np.array([[1.0]])

# our covariance matrix. After a second, we may stray by 16 bpm
p = np.array([[16.0]])

# Q grows the covariance over time
q = 0.01 / samples_per_second

data = {
    "time_of_previous_beat": 0,
    "bpm_to_find": 100.0,
}

observation_base_vector_tuples = np.arange(0.25, 4.25, 0.25)

transition_rate = (1.0 / 60) / samples_per_second


def find_z_in_vector_and_compute_r(observation_vector, bpm_estimate):
    idx = (np.abs(observation_vector - bpm_estimate)).argmin()
    return np.array([[observation_vector[idx]]]), np.array([[16.0]])


def z_activation(sample_in_beat, samples_per_beat):
    return random.choice([(sample_in_beat == 0 or sample_in_beat == random.choice([0, samples_per_beat / 2])), False])
    # return sample_in_beat == 0 or sample_in_beat == random.choice([0, samples_per_beat / 2])
    # return sample_in_beat == 0 or sample_in_beat == random.choice([0, samples_per_beat / 2]) or sample_in_beat == random.choice([0, samples_per_beat / 4])


def get_z_and_r(t, bpm_estimate):
    # data["bpm_to_find"] = data["bpm_to_find"] - 0.0001
    samples_per_beat = int(1 / ((data["bpm_to_find"] / 60) / samples_per_second))
    sample_in_beat = (t % samples_per_beat)
    if z_activation(sample_in_beat, samples_per_beat):
        time_since_last_beat = t - data["time_of_previous_beat"]
        data["time_of_previous_beat"] = t
        if time_since_last_beat:
            relative_bpm = 1 / (time_since_last_beat * transition_rate)
            return find_z_in_vector_and_compute_r(observation_base_vector_tuples * relative_bpm, bpm_estimate)


xs = []
zs = []
ps = []

start = timer()


for i in range(0, samples_per_second * 4000):
    p = p + q

    z_and_r = get_z_and_r(i, h.dot(x)[0, 0])
    if z_and_r is not None:
        z, r = z_and_r
        h_t = h.transpose()

        k_prime = p.dot(h_t).dot(np.linalg.inv(h.dot(p).dot(h_t) + r))
        x = x + k_prime.dot(z - h.dot(x))
        p = p - k_prime.dot(h.dot(p))
        zs.append(z[0, 0])
        xs.append(x[0, 0])

    ps.append(p[0, 0])

end = timer()
print(f"Time took kalman filter {str(end - start)}")

print(x[0,0])
print(p)

plt.figure()
plt.plot(xs,'b-')
# plt.plot(zs,'r-')
#
# plt.figure()
# plt.plot(ps[-4000:],'y-')

# plt.figure()
# plt.plot(ids,'y-')

plt.show()