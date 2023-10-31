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
    parser.add_argument("--save_path", required=False, type=str, default="./draw_results")
    
    args = parser.parse_args()
    slice_b = 1
    slice_e = None

    take_log10 = False
    
    for eps in args.epss:
        data = torch.load(args.result_path + "{}.pth".format(eps))
        
        data = data.numpy()
        if take_log10:
            data = np.log10(data)
        x = list(range(data.shape[0]-1))
        print(data[0])

        if slice_e is None:
            slice_e = len(x)
            
        x = x[slice_b: slice_e]
        data = data[slice_b: slice_e]
        
        
        plt.plot(x, data, label="$\epsilon$={}".format(eps))
    
    plt.legend(prop={'size': 8})
    plt.yticks(args.epss)
    
    ##### begin show or save
    if args.save:
        if args.pdf:
            plt.savefig(args.save_path, format='pdf')
        else:
            plt.savefig(args.save_path + '.png')
    else:
        plt.show()