import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# samples_per_second = 44100
samples_per_second = 200

transition_rate = (1.0 / 60) / samples_per_second

# bpm
x = np.array([[120.0]])

# our covariance matrix. After a second, we may stray by 16 bpm
p = np.array([[0.01]])

# Q grows the covariance over time
q = 16.0 / samples_per_second

bpm_to_find = 100.0
data = {
    "time_of_previous_beat": 0
}


def get_z(t):
    position_at_current_bpm = (t * transition_rate * bpm_to_find) % 1.0
    if position_at_current_bpm < transition_rate:
        time_since_last_beat = t - data["time_of_previous_beat"]
        data["time_of_previous_beat"] = t
        if time_since_last_beat:
            return np.array([[1.0]]) / (time_since_last_beat * transition_rate)


r = np.array([[1.0]])
h = np.array([[1.0]])

xs = []
zs = []
ps = []

start = timer()

for i in range(0, samples_per_second * 200):
    p = p + q

    z = get_z(i)
    if z is not None:
        h_t = h.transpose()

        k_prime = p.dot(h_t).dot(np.linalg.inv(h.dot(p).dot(h_t) + r))
        x = x + k_prime.dot(z - h.dot(x))
        p = p - k_prime.dot(h.dot(p))
        zs.append(z[0, 0])
    else:
        zs.append(bpm_to_find)

    ps.append(p[0, 0])
    xs.append(x[0, 0])

end = timer()
print(f"Time took kalman filter {str(end - start)}")

print(x[0,0])
print(p)

plt.figure()
plt.plot(xs,'b-')
plt.plot(zs,'r-')
#
# plt.figure()
# plt.plot(ps,'y-')

plt.show()