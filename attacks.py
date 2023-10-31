import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

mu = 0.1307
std = 0.3081

def normalize(X):
    return (X - mu)/std

def unnormalize(X):
    return X * std + mu


lower_limit, upper_limit = 0, 1

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
    
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, norm, device):
    delta = torch.zeros_like(X).to(device)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit-X, upper_limit-X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(normalize(X + delta))
        loss = F.cross_entropy(output, y)
        loss.backward()
        
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()
    return delta


def attack_pgd_records(model, X, y, epsilon, alpha, attack_iters, norm, device, avg_num):
    delta = torch.zeros_like(X).to(device)
    delta_list = []
    delta_list.append(delta.clone().detach().cpu())
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit-X, upper_limit-X)
    delta.requires_grad = True
    delta_list.append(delta.clone().detach().cpu())
    for _ in range(attack_iters):
        output = model(normalize(X + delta))
        loss = F.cross_entropy(output, y)
        loss.backward()
        
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()
        delta_list.append(delta.clone().detach().cpu())
    avg_tensors = delta_list[-avg_num:]
    stack_tensor = torch.stack(avg_tensors, dim=0)
    delta_star = stack_tensor.mean(dim=0)
    deltas = torch.stack(delta_list, dim=0)
    
    differences = (deltas - delta_star).view(deltas.shape[0], -1)
    if norm == "l_inf":
        norms = differences.abs().max(dim=1)[0]
    elif norm == "l_2":
        norms = torch.norm(differences, p=2, dim=1)
    
    return norms