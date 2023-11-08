from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import random
import math


t=np.arange(0, 0.06, 1e-5)
fig = plt.figure()
ax = fig.add_subplot(111)

################################################################
data_005 = np.load("./run_results/simu2/norm_list_0.5.npy")
data_1 = np.load("./run_results/simu2/norm_list_0.1.npy")
data_01 = np.load("./run_results/simu2/norm_list_0.01.npy")
data_02 = np.load("./run_results/simu2/norm_list_0.02.npy")
data_05 = np.load("./run_results/simu2/norm_list_0.05.npy")

# ax.plot_surface(x1, y1, z1, color='b')
ax.plot(t, data_005, label="$\\epsilon={0.5}$", color='purple')
ax.plot(t, data_1, label="$\\epsilon={0.1}$", color='orange')
ax.plot(t, data_05, label="$\\epsilon={0.05}$", color='r')
ax.plot(t, data_02, label="$\\epsilon={0.02}$", color='k')
ax.plot(t, data_01, label="$\\epsilon={0.01}$", color='b')





plt.yticks([-0, -2, - 4, -6, -8])
ax.set_yticklabels(
    ["$1$", "$10^{-2}$", "$10^{-4}$", "$10^{-6}$", "$10^{-8}$"],
    fontsize=14)
ax.set_xlabel('t') #x轴名称
plt.legend()

plt.show()
# plt.savefig("./run_results/simu2/ode_int"+ '.pdf', format='pdf')



# ax.set_xlim(0.9, 0.93)
# ax.set_ylim(0, 0.045)
# ax.set_zlim(0, 0.05)
# plt.show()
