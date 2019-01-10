import matplotlib.pyplot as plt
import numpy as np

data = np.reshape(np.fromfile('../logs/Imu3D.CheckPropagation.log', dtype=np.float64), (-1,27))

t = data[:, 0]
xhat = data[:, 1:11]
x = data[:, 11:21]
u = data[:, 21:27]

error = x - xhat


plt.figure()
plt.suptitle('Position')
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[:,i], label='x')
    plt.plot(t, xhat[:,i], label=r'\hat{x}')
    if i == 0:
        plt.legend()

plt.figure()
plt.suptitle('Velocity')
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[:,i+7], label='x')
    plt.plot(t, xhat[:, i+7], label=r'$\hat{x}$')
    if i == 0:
        plt.legend()

plt.figure()
plt.suptitle('Attitude')
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(t, x[:,i+3], label='x')
    plt.plot(t, xhat[:,i+3], label=r'\hat{x}')
    if i == 0:
        plt.legend()

plt.figure()
plt.suptitle('Input')
labels=['a_x', 'a_y', 'a_z', r'$\omega_x$', r'$\omega_{y}$', r'$\omega_{z}$']
for j in range(2):
    for i in range(3):
        plt.subplot(3, 2, i*2+1 + j)
        plt.plot(t, u[:,i+j*3], label=labels[i+j*3])
        plt.legend()

plt.figure()
plt.suptitle('Error State')
labels=[r'$p_x$', r'$p_y$', r'$p_z$',
        r'$v_x$', r'$v_y$', r'$v_z$',
        r'$q_x$', r'$q_y$', r'$q_z$']
for j in range(3):
    for i in range(3):
        plt.subplot(3, 3, i*3+1+j)
        plt.plot(t, error[:,i+j*3], label=labels[i+j*3])
        plt.legend()

plt.show()