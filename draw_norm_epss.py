import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # path related
    parser.add_argument("--result_path", required=False, type=str, default="./run_results/simu2/")
    parser.add_argument('--pdf', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    
    parser.add_argument("--epss", nargs='+', type=float, default=[0.1, 0.05, 0.02, 0.01, 0.005])
    parser.add_argument("--x0", required=False, type=float, choices=[None, 1], default=None)

    
    args = parser.parse_args()
    ax = plt.axes()
    take_log10 = True

    steps = 1000
    
    if args.x0 is not None:
        args.x0 = [1, 0]
        steps = 800
    
    if args.x0 is None:
        filename = "_norm_None"
    else:
        filename = "_norm_{}".format(args.x0)
        
    for eps in args.epss:
        
        data = np.load(args.result_path + filename + "_{}.npy".format(eps))
        if steps <= len(data):
            data = data[:steps]
        x = list(range(data.shape[0]))
        
        if take_log10:
            data = np.log10(data)
        if eps == 0.01:
            plt.plot(x, data, label="$\\epsilon={}$".format(eps), color='b')
        elif eps == 0.5:
            plt.plot(x, data, label="$\\epsilon={}$".format(eps), color='purple')
        elif eps == 0.1:
            plt.plot(x, data, label="$\\epsilon={}$".format(eps), color='orange')
        elif eps == 0.05:
            plt.plot(x, data, label="$\\epsilon={}$".format(eps), color='r')
        elif eps == 0.02:
            plt.plot(x, data, label="$\\epsilon={}$".format(eps), color='k')
        elif eps == 0.005:
            plt.plot(x, data, label="$\\epsilon={}$".format(eps), color='green')
        else:
            plt.plot(x, data, label="$\\epsilon={}$".format(eps))
    
    plt.yticks([-0, -2,- 4, -6, -8, -10])
    ax.set_yticklabels(
        ["$0$", "$10^{-2}$", "$10^{-4}$", "$10^{-6}$", "$10^{-8}$", "$10^{-10}$"],
        fontsize=14)

    plt.legend()
    
    if args.save:
        if args.pdf:
            plt.savefig(filename + '.pdf', format='pdf')
        else:
            plt.savefig(filename + '.png')
    else:
        plt.show()