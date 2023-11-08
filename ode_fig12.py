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
    parser.add_argument("--eps", required=False, type=float, default=0.1)
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
    return np.array([-sigma**(-2)*(x + y)*x + 1, -sigma**(-2)*(x + y)*y + 1])
t_end= 0.06
t=np.arange(0, t_end, args.step_size) #建立自变量序列（也就是时间点）
x_start = args.eps * np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
x_end=(args.eps*math.sqrt(2)/2, args.eps*math.sqrt(2)/2)
sol1=odeint(lorenz,x_start, t) #第一个初值问题求解

sol2 = []
for i in range(0, int(t_end/args.step_size)+1):
    sol2.append(math.log10(math.sqrt((sol1[i,0]-args.eps*math.sqrt(2)/2)**2+ (sol1[i,1]-args.eps*math.sqrt(2)/2)**2)))


norm_path = os.path.join(args.save_path, "ode_path_{}".format(args.eps))
norm_list = os.path.join(args.save_path, "odenorm_list_{}".format(args.eps))


np.save(norm_path, sol1)
np.save(norm_list, sol2)




