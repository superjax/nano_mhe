import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(linewidth=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


data = np.reshape(np.fromfile('../logs/Imu3d.CheckDynamics.log', dtype=np.float64), (-1,46))

t = data[:,0]
y = data[:,1:11]
yhat = data[:,11:21]
dy = data[:,21:30]
y_p_dy = data[:,30:40]
u = data[:, 40:]
error = (y - y_p_dy)

plt.figure()
plt.suptitle('Position')
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, y[:,i], label='x')
    plt.plot(t, yhat[:,i], label=r'\hat{x}')
    plt.plot(t, y_p_dy[:, i], label=r'$\hat{x} + \delta x$')
    if i == 0:
        plt.legend()

plt.figure()
plt.suptitle('Velocity')
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, y[:,i+3], label='x')
    plt.plot(t, yhat[:,i+3], label=r'\hat{x}')
    plt.plot(t, y_p_dy[:, i+3], label=r'$\hat{x} + \delta x$')
    if i == 0:
        plt.legend()

plt.figure()
plt.suptitle('Attitude')
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(t, y[:,i+6], label='x')
    plt.plot(t, yhat[:,i+6], label=r'\hat{x}')
    plt.plot(t, y_p_dy[:, i+6], label=r'$\hat{x} + \delta x$')
    if i == 0:
        plt.legend()

plt.figure()
plt.suptitle('Error State')
labels=[r'$p_x$', r'$p_y$', r'$p_z$',
        r'$v_x$', r'$v_y$', r'$v_z$',
        r'$q_x$', r'$q_y$', r'$q_z$']
for j in range(3):
    for i in range(3):
        plt.subplot(3, 3, i+1+j*3)
        plt.plot(t, dy[:,i+j*3], label=labels[i+j*3])
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
plt.suptitle("Error")
for i in range(error.shape[1]):
    plt.plot(t, error[:,i])
plt.plot(t, 5e-6*t*t, '--')
plt.plot(t, -5e-6*t*t, '--')

plt.show()

