import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_mnist import smallCNN_MNIST
from torchvision.datasets import MNIST
import random
import numpy as np
import torchvision.transforms as transforms
from utils import set_seed
from attacks import normalize, unnormalize, attack_pgd_records

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # path related
    parser.add_argument("--model_dir", required=False, type=str, default=None)
    parser.add_argument("--data_dir", required=False, type=str, default="/home/zx/data")
    parser.add_argument("--save_path", required=False, type=str, default="./run_results/")
    
    # tuning parameters
    parser.add_argument("--eps", required=False, type=float, default=0.01)
    parser.add_argument("--step_size", required=False, type=float, default=0.000002)
    parser.add_argument("--step_num", required=False, type=int, default=10000)
    parser.add_argument("--avg_num", required=False, type=int, default=5, help="The number of last iters to average as x_star")
    parser.add_argument("--seed", required=False, type=int, default=1)
    parser.add_argument("--input_idx", required=False, type=int, default=0)
    parser.add_argument("--norm", required=False, type=str, default="l_2")
    
    args = parser.parse_args()
    
    ##############
    args.step_size = 2*args.eps / args.step_num
    ##############
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = smallCNN_MNIST().to(device)
    if args.model_dir is not None:
        model.load_state_dict(torch.load(args.model_dir))
        
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    trainset = MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

    testset = MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)
    
    x, y = next(iter(testloader))
    x = x[args.input_idx: args.input_idx+1]
    y = y[args.input_idx: args.input_idx+1]
    x = x.to(device)
    y = y.to(device)
    
    model.eval()
    
    norms = attack_pgd_records(model, x, y, args.eps, args.step_size, args.step_num, args.norm, device, args.avg_num)
    
    torch.save(norms, args.save_path + "{}.pth".format(args.eps))
    