import argparse
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import vision_transformer as vits
from loguru import logger
from torch.nn import functional as F
from config import *


class WeightedGamma(nn.Module):
    def __init__(self, args):
        super(WeightedGamma, self).__init__()

        self.weight1 = nn.Parameter(torch.ones(1, 1).cuda())
        self.weight2 = nn.Parameter(torch.ones(1, 1).cuda())
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y):
        x = x.cuda()
        y = y.cuda()
        w1 = self.sigmoid(self.weight1)
        w2 = self.sigmoid(self.weight2)
        gamma = w1 / (w1 + w2)
        
        return (1-gamma) * x + gamma * y


class AverageFusion(nn.Module):
    def __init__(self):
        super(AverageFusion, self).__init__()
        self.gamma = None 
    def forward(self, x, y):
        x = x.cuda()
        y = y.cuda()
        fused = (x + y) / 2
        
        return fused

class ConcatFusion(nn.Module):
    def __init__(self, input_dim):
        super(ConcatFusion, self).__init__()
        self.input_dim = input_dim
        self.gamma = None 
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.ReLU()  
        ).cuda()

    def forward(self, x, y):
        x = x.float().cuda() # (128,768) batch_size, input_dim
        y = y.float().cuda()
        concatenated = torch.cat((x, y), dim=1)
        fused = self.fusion(concatenated)

        return fused    


def check_parms(model):
    for name, parms in model.named_parameters():
        print(f'{name} : {parms}')      
            
         
def check_is_grad(model,optim_params=None):
    if optim_params !=None:
        for name, parms in model.named_parameters():
                    # print(f'-->name:{name} -->grad_requirs:{parms.requires_grad}')
                    # print(f'-->weight{torch.mean(parms.data)} -->grad_value:{torch.mean(parms.grad)}')
            if name in optim_params:
                # print(f'-->weight{parms.data} -->grad_value:{parms.grad}')
                print(f'-->name:{name} -->grad_requirs:{parms.requires_grad} -->grad_value:{parms.grad}')
    else:
        for name, parms in model.named_parameters():
            print(f'-->name:{name} -->grad_requirs:{parms.requires_grad} -->grad_value:{parms.grad}')
           
def change_need_grad(model,optim_params=None):
    for name, parms in model.named_parameters():
        if name in optim_params:
            parms.requires_grad = True
        else:
            parms.requires_grad = False #名字不在array中的全部参数冻结

    
def get_optim_params(model_name: str):
    return ['visual.transformer.resblocks.23.attn.in_proj_weight',
                'visual.transformer.resblocks.23.attn.in_proj_bias',
                'visual.transformer.resblocks.23.attn.out_proj.weight',
                'visual.transformer.resblocks.23.attn.out_proj.bias',
                'visual.transformer.resblocks.23.ln_1.weight',
                'visual.transformer.resblocks.23.ln_1.bias',
                'visual.transformer.resblocks.23.mlp.c_fc.weight',
                'visual.transformer.resblocks.23.mlp.c_fc.bias',
                'visual.transformer.resblocks.23.mlp.c_proj.weight',
                'visual.transformer.resblocks.23.mlp.c_proj.bias',
                'visual.transformer.resblocks.23.ln_2.weight',
                'visual.transformer.resblocks.23.ln_2.bias']



get_dataset_num = {
    'cifar10': pretrained_cifar10_num,
    'cifar100': pretrained_cifar100_num,
    'imagenet_100': pretrained_imagenet_100_num,
    'cub': pretrained_cub_num,
    'aircraft': pretrained_cub_num
}
get_dataset_name = {
    'cifar10': pretrained_cifar10_name,
    'cifar100': pretrained_cifar100_name,
    'imagenet_100': pretrained_imagenet_100_name,
    'cub': pretrained_cub_name,
    'aircraft': pretrained_cub_name
}


def get_pretrained_model(args):
    pretrained_model_num = get_dataset_num[args.dataset_name]
    pretrained_model_name = get_dataset_name[args.dataset_name]
    return pretrained_save_path + pretrained_model_num + pretrained_model_name 