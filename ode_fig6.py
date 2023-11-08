from scipy.integrate import odeint
import numpy as np
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math

def project_delta(eps: float, delta: np.ndarray):
    l2_norm = np.linalg.norm(delta, ord=2)
    if l2_norm > eps:
        delta = delta * eps / l2_norm
    return delta


def lorenz(w,t): #定义微分方程组
	sigma=0.1
	x,y,z=w
	return np.array([sigma ** (-2) * (x *(x +1) + y ** 2 + z ** 2) * x - (x + 1), sigma ** (-2) * (x *(x +1) + y ** 2 + z ** 2) * y - y,
                     sigma ** (-2) * (x *(x +1) + y ** 2 + z ** 2) * z - z])
t=np.arange(0, 100, 0.01) #建立自变量序列（也就是时间点）
x_01 = 1 * np.array([math.cos(math.radians(150)), math.sin(math.radians(150)), math.sin(math.radians(135))])
x_0 = project_delta(0.1, x_01)
# x_0=(0.1, 0, 0)
sol1=odeint(lorenz,x_0 , t) #第一个初值问题求解

#画图代码 （可忽略）
plt.rc('font',size=16); plt.rc('text',usetex=False)
#第一个图的各轴的定义
ax1=plt.subplot(111,projection='3d')
ax1.plot(sol1[:,0], sol1[:,1], sol1[:,2], 'r', linewidth=3.0)
# print(sol1)

norm_path = "./run_results/simu2/ode_path"
np.save(norm_path, sol1)

# ax1.set_xlabel('$x$');ax1.set_ylabel('$y$');ax1.set_zlabel('$z$')
# plt.show()

