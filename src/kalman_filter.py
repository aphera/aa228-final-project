import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# samples_per_second = 44100
samples_per_second = 200

transition_rate = (1.0 / 60) / samples_per_second

# bpm
x = 120.0

# our covariance matrix. After a second, we may stray by 16 bpm
p = 16.0 / samples_per_second

# Q grows the covariance over time
# TODO refine this value
q = 16.0


r = 1.0 / samples_per_second

bpm_to_find = 100.0

time_delta = 1.0 / samples_per_second

data = {
    "time_of_previous_beat": 0
}

def get_z(t):
    position_at_current_bpm = (t * transition_rate * bpm_to_find) % 1.0
    if position_at_current_bpm < transition_rate:
        time_since_last_beat = t - data["time_of_previous_beat"]
        data["time_of_previous_beat"] = t
        if time_since_last_beat:
            return 1 / (time_since_last_beat * transition_rate)
    return 0
    # return observation_base_vector * position_at_current_bpm


def get_h(s):
    return s
    # return observation_base_vector * s


xs = []
ts = []
zs = []
ps = []

start = timer()

for i in range(0, samples_per_second * 30):
    p = p + q

    z = get_z(i)
    if z:
        h = get_h(x)

        k_prime = p / (p + r)
        x = x + k_prime * (z - x)
        p = (1 - k_prime) * p

        # k_prime = p.dot(h_t).dot(np.linalg.inv(h.dot(p).dot(h_t) + r))
        # x = (x + k_prime.dot(z - h.dot(x)))
        # p = p - k_prime.dot(h.dot(p))
        ps.append(p)
        xs.append(x)
        zs.append(bpm_to_find)

end = timer()
print(f"Time took kalman filter {str(end - start)}")

print(x)
print(p)

# plt.figure()
# plt.plot(ts,'g-')

# plt.figure()
# plt.plot(xs,'b-')
# plt.plot(zs,'r-')
# # plt.plot(xs[-400:],'b-')
# # plt.plot(zs[-400:],'r-')
#
plt.figure()
plt.plot(ps,'y-')

plt.show()