import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

samples_per_second = 44100

# position, samples per beat
x = np.array([0.0, 0.0])
# Our transition function
f = np.array(
    [[1.0, (1.0 / 60) / samples_per_second],
     [0.0, 1.0]]
)
f_t = f.transpose()
# our convariance matrix
p = np.array(
    [[0.0, 1.0],
     [0.0, 0.0]]
)
# Q grows the covariance over time
q = np.array(
    [[0.001, 0.001],
     [0.001, 0.001]]
)

# Our observation function
h = np.array([[1.2,2],[3,4],[5,6]])
h_t = h.transpose()
# Means of observations
z = np.array([5.5,2,3])
# The convariance of the observations
r = np.array([[1,2,3],[4,5,6],[7,8,9]])


start = timer()

for i in range(0, samples_per_second):
    x = f.dot(x)
    p = f.dot(p).dot(f_t) + q

    k_prime = p.dot(h_t).dot(np.linalg.inv(h.dot(p).dot(h_t) + r))
    x = x + k_prime.dot(z - h.dot(x))
    p = p - k_prime.dot(h.dot(p))

end = timer()
print(f"Time took kalman filter {str(end - start)}")

print(x)
print(p)

# plt.figure()
# plt.plot(a,'b-')
# plt.plot(b,'r-')
# plt.show()
