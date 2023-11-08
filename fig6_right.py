from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import random
import math



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

t = np.linspace(0, np.pi * 2, 100)
s = np.linspace(0, np.pi, 100)

t, s = np.meshgrid(t, s)
x = 1+0.1*np.cos(t) * np.sin(s)
y = 0.1*np.sin(t) * np.sin(s)
z = 0.1*np.cos(s)
# z[>0]=0  # 截取球体的下半部分
ax = plt.subplot(111, projection='3d')
# ax = plt.subplot(121, projection='3d')
# ax.plot_wireframe(x, y, z)
# ax = plt.subplot(122, projection='3d')
# ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
# ax = plt.subplot(122, projection='3d')
ax.set_xlabel('x axis') #x轴名称
ax.set_ylabel('y axis') #y轴名称
ax.set_zlabel('z axis') 
ax.plot_surface(x, y, z, alpha=0.2, rstride=1, cstride=1, color='w')
# plt.gca().set_box_aspect((2,2,1)) #设置坐标比例时2：2：1


################################################################

data_1 = np.load("./run_results/simu2/_path_[1, 0, 0]_1.0.npy")
data_05 = np.load("./run_results/simu2/_path_[1, 0, 0]_0.5.npy")
data_02 = np.load("./run_results/simu2/_path_[1, 0, 0]_0.2.npy")
data_01 = np.load("./run_results/simu2/_path_[1, 0, 0]_0.1.npy")
data_001= np.load("./run_results/simu2/_path_[1, 0, 0]_0.05.npy")
# ax.plot_surface(x1, y1, z1, color='b')
ax.plot3D(data_1[:, 0], data_1[:, 1], data_1[:, 2], label='PGD $\eta=1$', color='orange', linewidth=1)
ax.plot3D(data_05[:, 0], data_05[:, 1], data_05[:, 2], label='PGD $\eta=0.5$', color='purple', linewidth=1)
ax.plot3D(data_02[:, 0], data_02[:, 1], data_02[:, 2], label='PGD $\eta=0.2$', color='yellow', linewidth=1)
ax.plot3D(data_01[:, 0], data_01[:, 1], data_01[:, 2], label='PGD $\eta=0.1$', color='green', linewidth=1)
ax.plot3D(data_001[:, 0], data_001[:, 1], data_001[:, 2], label='PGD $\eta=0.05$', color='blue', linewidth=1)

data_ode=np.load("./run_results/simu2/ode_path.npy")
ax.plot3D(data_ode[:, 0] + 1, data_ode[:, 1], data_ode[:, 2], label='ODE trajectory', color='red', linewidth=1)

ax.scatter3D(data_ode[0, 0] + 1, data_ode[0, 1], data_ode[0, 2], color='red')
ax.text(data_ode[0, 0] + 1, data_ode[0, 1], data_ode[0, 2], '  $\delta_0$', ha='left')
ax.scatter3D(0.9 ,0 ,0, color='red')
ax.text(0.9 ,0 ,0, '   $\delta^\Delta$', ha='right')
ax.scatter3D(1 ,0 ,0, color='black')
ax.text(1 ,0 ,0, ' O', ha='left')
# print(data_ode[0, 0] + 1, data_ode[0, 1], data_ode[0, 2])

ax.set_xlabel('$\delta^{(1)}$') # 画出坐标轴\delta=(0.9 ,0 ,0)
ax.set_ylabel('$\delta^{(2)}$')
ax.set_zlabel('$\delta^{(3)}$')
# ax.legend(bbox_to_anchor=(0.2, 0.8), prop = {'size':9})

# plt.show()
# plt.savefig("./run_results/simu2/ode_int"+ '.pdf', format='pdf')



ax.set_xlim(0.9, 0.93)
ax.set_ylim(0, 0.045)
ax.set_zlim(0, 0.05)
plt.show()
