import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # path related
    parser.add_argument("--result_path", required=False, type=str, default="./run_results/")
    parser.add_argument('--pdf', type=int, default=0)
    parser.add_argument('--save', type=int, default=1)
    
    parser.add_argument("--x0", required=False, type=float, nargs='+', choices=[None, [1, 0]], default=[1, 0])
    # parser.add_argument("--x0", required=False, type=float, nargs='+', choices=[None, [1, 0]], default=None)
    
    args = parser.parse_args()
    ax = plt.axes()
    
    if args.x0 is None:
        filename = "simu_norm_None"
    else:
        filename = "simu_norm_{}".format(args.x0)
        
    data = np.load(args.result_path + filename + ".npy")
    x = list(range(data.shape[0]))
    
    take_log10 = True
    if take_log10:
        data = np.log10(data)
    
    plt.plot(x, data)
    
    if args.x0 is None:
        plt.yticks([-2, -4, -6, -8, -10])
        ax.set_yticklabels(["$10^{-2}$", "$10^{-4}$", "$10^{-6}$", "$10^{-8}$", "$10^{-10}$"], fontsize=14)
    else:
        plt.yticks([-20, -40, -60, -80, -100, -120, -140, -160])
        ax.set_yticklabels(["$10^{-20}$", "$10^{-40}$", "$10^{-60}$", "$10^{-80}$", "$10^{-100}$", "$10^{-120}$", "$10^{-140}$", "$10^{-160}$"], fontsize=14)

    if args.save:
        if args.pdf:
            plt.savefig(filename, format='pdf')
        else:
            plt.savefig(filename + '.png')
    else:
        plt.show()