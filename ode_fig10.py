from scipy.integrate import odeint
import numpy as np
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path related
    parser.add_argument("--save_path", required=False, type=str, default="./run_results/simu2")

    # tuning parameters
    parser.add_argument("--eps", required=False, type=float, default=0.01)
    parser.add_argument("--step_size", required=False, type=float, default=1e-5)
    parser.add_argument("--step_num", required=False, type=int, default=2000)
    parser.add_argument("--seed", required=False, type=int, default=2)

    parser.add_argument("--x0", required=False, type=float, nargs='+', choices=[None, [1, 0]], default=[1, 0])
    args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

def project_delta(eps: float, delta: np.ndarray):
    l2_norm = np.linalg.norm(delta, ord=2)
    if l2_norm > eps:
        delta = delta * eps / l2_norm
    return delta


def lorenz(w, t): #定义微分方程组
    sigma=args.eps
    x,y =w
    return np.array([sigma**(-2)*x*2*(x+1)*x-2*(x+1), sigma**(-2)*2*(x +1)*x*y])

t_end= 0.06
t=np.arange(0, t_end, args.step_size) #建立自变量序列（也就是时间点）
x_start = args.eps * np.array([math.cos(math.radians(150)), math.sin(math.radians(150))])
x_end=(1-args.eps, 0)
sol1=odeint(lorenz,x_start, t) #第一个初值问题求解

sol2 = []

for i in range(0, int(t_end/args.step_size)+1):
    sol2.append(math.log10(math.sqrt((sol1[i,0]+args.eps)**2+ sol1[i,1]**2)))


norm_path = os.path.join(args.save_path, "ode_path_{}".format(args.eps))
norm_list = os.path.join(args.save_path, "norm_list_{}".format(args.eps))


np.save(norm_path, sol1)
np.save(norm_list, sol2)


# #画图代码 （可忽略）
# plt.rc('font',size=16); plt.rc('text',usetex=False)
# #第一个图的各轴的定义
# ax1=plt.subplot(111)
# ax1.plot(t, sol2, 'r', linewidth=3.0, color='blue')
# print(sol2)
#
# plt.yticks([-2,- 4, -6, -8])
# ax = plt.axes()
# ax.set_yticklabels(
#         [ "$10^{-2}$", "$10^{-4}$", "$10^{-6}$", "$10^{-8}$"],
#         fontsize=14)
# ax.set_xlabel('t') #x轴名称
# plt.show()



############################################

# #画图代码 （可忽略）
# plt.rc('font',size=16); plt.rc('text',usetex=False)
# #第一个图的各轴的定义
# ax1=plt.subplot(111)
# ax1.plot(sol1[:,0]+1, sol1[:,1] , 'r', linewidth=3.0, color='blue')
# print(sol1)
#
# theta = np.linspace(0, 2 * np.pi, 200)
# circ_x = args.eps * np.cos(theta) + 1
# circ_y = args.eps * np.sin(theta) + 0
# plt.plot(circ_x, circ_y, color='red', linewidth=1)
# ax1.set_aspect('equal', adjustable='box')
# ax1.set_xlabel('$x$');ax1.set_ylabel('$y$')
# plt.show()

