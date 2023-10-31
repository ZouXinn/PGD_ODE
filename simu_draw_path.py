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
    parser.add_argument("--save_path", required=False, type=str, default="./draw_results")
    
    parser.add_argument("--eps", required=False, type=float, default=0.1)
    parser.add_argument("--x0", required=False, type=float, nargs='+', choices=[None, [1, 0]], default=[1, 0])
    # parser.add_argument("--x0", required=False, type=float, nargs='+', choices=[None, [1, 0]], default=None)
    
    args = parser.parse_args()
    
    
    if args.x0 is None:
        filename = "simu_path_None"
        x0=(0.05,0)
    else:
        filename = "simu_path_{}".format(args.x0)
        x0=tuple(args.x0)
        
    data = np.load(args.result_path + filename + ".npy")
    
    
    padding = 0.01
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    
    
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_x = args.eps * np.cos(theta) + x0[0]
    circ_y = args.eps * np.sin(theta) + x0[1]
    plt.plot(circ_x, circ_y, color='red', linewidth=1)
    
    x = data[:,0]
    y = data[:,1]
    
    plt.plot(x, y)
    
    # plt.xlim(x0[0]-args.eps-padding,x0[0]+args.eps+padding)
    # plt.ylim(x0[1]-args.eps-padding,x0[1]+args.eps+padding)
    
    
    if args.save:
        if args.pdf:
            plt.savefig(filename, format='pdf')
        else:
            plt.savefig(filename + '.png')
    else:
        plt.show()