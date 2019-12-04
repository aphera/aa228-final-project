import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# samples_per_second = 44100
samples_per_second = 200

transition_rate = (1.0 / 60) / samples_per_second

bpm = 120.0
bpm_estimate = 120.0

# position within beat
x = np.array([[0.0]])


learning_rate = 1 / (samples_per_second)
# Our transition function
def get_f():
    return bpm * transition_rate
    # return bpm_estimate * transition_rate FIXME
# our covariance matrix
p = np.array([[0.0]])

# Q grows the covariance over time
# TODO refine this value
q = 1.0


observation_base_vector = np.array([[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]]).transpose()

r_shape = observation_base_vector.dot(observation_base_vector.T).shape
r = np.ones(r_shape) + np.random.rand(*r_shape)

bpm_to_find = 120.0


def get_z(t):
    position_at_current_bpm = ((t + 1) * transition_rate * bpm_to_find)
    return observation_base_vector * position_at_current_bpm


def get_h(s):
    return observation_base_vector * s


xs = []
ts = []
zs = []
ps = []

start = timer()

for i in range(0, samples_per_second):
    prev_x = x[0,0]
    f = get_f()
    x = x + f
    p = p + q

    x[0,0] = x[0,0]
    z = get_z(i)
    h = get_h(x[0,0])
    h_t = h.transpose()



    k_prime = p.dot(h_t).dot(np.linalg.inv(h.dot(p).dot(h_t) + r))
    x = (x + k_prime.dot(z - h.dot(x)))
    p = p - k_prime.dot(h.dot(p))
    ps.append(p[0,0])
    xs.append(x[0,0])
    ts.append(bpm_estimate)
    zs.append(z[2])

    new_x = x[0,0]
    # Update according to learning rate
    current_bpm_in_this_update = (new_x - prev_x) / transition_rate
    bpm_estimate += learning_rate * (current_bpm_in_this_update - bpm_estimate)

end = timer()
print(f"Time took kalman filter {str(end - start)}")

print(x)
print(p)

plt.figure()
plt.plot(ts,'g-')

# plt.figure()
# plt.plot(xs,'b-')
# plt.plot(zs,'r-')
# # plt.plot(xs[-400:],'b-')
# # plt.plot(zs[-400:],'r-')
#
# plt.figure()
# plt.plot(ps,'y-')



plt.show()