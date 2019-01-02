import matplotlib.pyplot as plt
import numpy as np


data = np.reshape(np.fromfile('../logs/nano_mhe.Optimize.log', dtype=np.float64), (10,-1))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure()
plt.subplot(311)
plt.plot(data[0,:], data[1,:], '--', label=r'$x$')
plt.plot(data[0,:], data[5,:], label=r'$\hat{x}$')
# plt.plot(data[0,:], data[3,:], label=r'$\int\int\bar{a}$')
plt.plot(data[0,:], data[9,:],'.', label=r'$\bar{x}$')
plt.xlabel("t(s)")
plt.ylabel("m")
plt.legend()

plt.subplot(312)
plt.plot(data[0,:], data[2,:], '--', label=r'$v$')
plt.plot(data[0,:], data[6,:], label=r'$\hat{v}$')
# plt.plot(data[0,:], data[4,:], label=r'$\int\bar{a}$')
plt.xlabel("t(s)")
plt.ylabel("m/s")
plt.legend()

plt.subplot(313)
plt.plot(data[0,:], data[7,:], '--', label=r'$b$')
plt.plot(data[0,:], data[8,:], label=r'$\hat{b}$')
plt.xlabel("t(s)")
plt.ylabel("m/s^2")
plt.legend()

plt.show()