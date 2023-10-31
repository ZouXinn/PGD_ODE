import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # path related
    parser.add_argument("--result_path", required=False, type=str, default="./run_results/")
    parser.add_argument('--epss', type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument('--pdf', type=int, default=0)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument("--save_path", required=False, type=str, default="./draw_local_results")
    
    args = parser.parse_args()
    slice_b = 8000
    slice_e = 8500

    take_log10 = True
    ax = plt.axes()
    for eps in args.epss:
    
        data = torch.load(args.result_path + "{}.pth".format(eps))
        
        data = data.numpy()
        if take_log10:
            data = np.log10(data)
        x = list(range(data.shape[0]))

        if slice_e is None:
            slice_e = len(x)
            
        x = x[slice_b: slice_e]
        data = data[slice_b: slice_e]
        
        
        plt.plot(x, data, label="eps={}".format(eps))
    
    plt.yticks([-2, -3])
    ax.set_yticklabels(["$10^{-2}$", "$10^{-3}$"], fontsize=14)
    
    ##### begin show or save
    if args.save:
        if args.pdf:
            plt.savefig(args.save_path, format='pdf')
        else:
            plt.savefig(args.save_path + '.png')
    else:
        plt.show()