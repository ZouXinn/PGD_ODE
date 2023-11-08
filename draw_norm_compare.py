import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path related
    parser.add_argument("--result_path", required=False, type=str, default="./run_results/simu2/")
    parser.add_argument('--pdf', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)

    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--x0", required=False, type=float, choices=[None, 1], default=None)

    args = parser.parse_args()
    ax = plt.axes()
    take_log10 = True
    
    if args.x0 is not None:
        args.x0 = [1, 0]


    if args.x0 is None:
        steps = 2000
        filename = "_norm_None"
        data = np.load(args.result_path + filename + "_{}.npy".format(args.eps))

        if steps <= len(data):
            data = data[1:steps]

        x = list(range(data.shape[0]))
        x = x[1:]
        data = data[1:]
        y_inv = 1/np.array(x)
        y_exp = np.exp2(- np.exp2(x))
        # y_exp = np.power(1.001, - np.exp2(x))

        if take_log10:
            data = np.log10(data)
            y_inv = np.log10(y_inv)
            y_exp = np.log10(y_exp)

        exp_num = 5
        plt.plot(x, data, label="linear", color='b')
        plt.plot(x, y_inv, label="sublinear", color='r')
        plt.plot(x[:exp_num], y_exp[:exp_num], label="superlinear", color='g')

        plt.yticks([-0, -2, -4, -6, -8, -10])
        ax.set_yticklabels(
            ["$0$", "$10^{-2}$", "$10^{-4}$", "$10^{-6}$", "$10^{-8}$", "$10^{-10}$"],
            fontsize=14)

        plt.legend()

        if args.save:
            if args.pdf:
                plt.savefig('rates.pdf', format='pdf')
            else:
                plt.savefig('rates.png')
        else:
            plt.show()
    else:
        steps = 800
        filename = "_norm_{}".format(args.x0)
        data = np.load(args.result_path + filename + "_{}.npy".format(args.eps))

        if steps <= len(data):
            data = data[1:steps]

        x = list(range(data.shape[0]))
        if take_log10:
            data = np.log10(data)

        plt.plot(x, data, color='b')

        plt.yticks([-2, -4, -6])
        ax.set_yticklabels(
            ["$10^{-2}$", "$10^{-4}$", "$10^{-6}$"],
            fontsize=14)
        plt.legend()

        if args.save:
            if args.pdf:
                plt.savefig('rates.pdf', format='pdf')
            else:
                plt.savefig('rates.png')
        else:
            plt.show()