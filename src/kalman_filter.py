import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# samples_per_second = 44100
samples_per_second = 200

transition_rate = (1.0 / 60) / samples_per_second

# position, beats per minute
x = np.array([0.0, 120.0])
# Our transition function
f = np.array(
    [[1.0, transition_rate],
     [0.0, 1.0]]
)
f_t = f.transpose()
# our covariance matrix
p = np.array(
    [[0.0, 0.0],
     [0.0, 0.0]]
)
# Q grows the covariance over time
# Let's assume after a second we may have drifted by 16bpm
q = np.array(
    [[1.0, 0.0],
     [0.0, 16.0]]
) / samples_per_second

r = np.array(
    [[1.0, 0.0],
     [0.0, 16.0]]
) / (samples_per_second)


time_delta = r[0,0]

bpm_to_find = 200.0


def get_z(i):
    position_at_100_bpm = i * transition_rate * bpm_to_find
    return np.array([position_at_100_bpm % 1.0, bpm_to_find])


def get_h(s):
    return np.array(
        [[1.0, 0.0],
         [0.0, 1.0]]
    )

start = timer()

xs = []
ts = []
zs = []

for i in range(0, samples_per_second * 2):
    x = f.dot(x)
    p = f.dot(p).dot(f_t) + q

    x[0] = x[0] % 1
    z = get_z(i)
    h = get_h(x)
    h_t = h.transpose()

    xs.append(x[0])
    ts.append(x[1])
    zs.append(z[0])

    k_prime = p.dot(h_t).dot(np.linalg.inv(h.dot(p).dot(h_t) + r))
    x = (x + k_prime.dot(z - h.dot(x)))
    p = p - k_prime.dot(h.dot(p))

end = timer()
print(f"Time took kalman filter {str(end - start)}")

print(x)
print(p)

plt.figure()
# plt.plot(xs,'b-')
# plt.plot(zs,'r-')
plt.plot(ts,'r-')
plt.show()
