import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import math

a = 2
b = 1

def f(x: np.ndarray):
    return - (a*x[0]^2 + b*x[1]^2) 

def f_prime(x: np.ndarray):
    return -2 * np.array([a,b]) * x

def project_delta(eps: float, delta: np.ndarray):
    l2_norm = np.linalg.norm(delta, ord=2)
    if l2_norm > eps:
        delta = delta * eps / l2_norm
    return delta


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pdf', type=int, default=0)
    parser.add_argument('--save', type=int, default=1)
    
    # path related
    parser.add_argument("--save_path", required=False, type=str, default="./run_results/simu2")
    
    # tuning parameters
    parser.add_argument("--eps", required=False, type=float, default=0.1)
    parser.add_argument("--step_size", required=False, type=float, default=5e-5)
    parser.add_argument("--step_num", required=False, type=int, default=2000)
    parser.add_argument("--seed", required=False, type=int, default=2)
    
    parser.add_argument("--x0", required=False, type=float, nargs='+', choices=[None, [1, 0]], default=[1, 0])
    # parser.add_argument("--x0", required=False, type=float, nargs='+', choices=[None, [1, 0]], default=None)

    args = parser.parse_args()
    
    ##############
    # args.step_size = 100*args.eps / args.step_num
    ##############
    
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.x0 is not None:
        x0 = np.array(args.x0)
        x_star = np.array([1-args.eps, 0])
        # delta = np.array([1.2/20, 1.3/20])
        delta = args.eps * np.array([math.cos(math.radians(150)), math.sin(math.radians(150))])
    else:
        x0 = np.array([0.05, 0])
        x_star = np.array([0, 0])
        # delta = np.array([1.2/20, 1.3/20])
        delta = np.array([1.2/20, -1.3/20])
        
    
    
    # delta = np.random.normal(0, args.eps, size=(2))
    delta = project_delta(args.eps, delta)
    
    path_list = []
    norm_list = []
    
    path_list.append(x0+delta)
    norm_list.append(np.linalg.norm(x0+delta-x_star, ord=2))
    
    for step in range(args.step_num):
        gradient = f_prime(x0 + delta)
        norm = np.linalg.norm(gradient, ord=2)
        delta += args.step_size * gradient # / norm
        delta = project_delta(args.eps, delta)
        path_list.append(x0+delta)
        norm_list.append(np.linalg.norm(x0+delta-x_star, ord=2))
    
    # print(path_list)
    # print(norm_list)

    
    if args.x0 is None:
        norm_path = args.save_path + "_norm_None"
        path_path = args.save_path + "_path_None"
    else:
        norm_path = os.path.join(args.save_path, "_norm_{}_{}".format(args.x0, args.eps))
        path_path = os.path.join(args.save_path, "_path_{}_{}".format(args.x0, args.eps))
        
    
    np.save(norm_path, norm_list)
    np.save(path_path, path_list)
    
    
    
    
    
    # if args.save:
    #     if args.x0 is None:
    #         save_path = args.save_path + "_None"
    #     else:
    #         save_path = args.save_path + "_{}".format(args.x0)
            
    #     if args.pdf:
    #         plt.savefig(save_path, format='pdf')
    #     else:
    #         plt.savefig(save_path + '.png')
    # else:
    #     plt.show()
    