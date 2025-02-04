from torchvision.datasets import CIFAR10,CIFAR100
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from loguru import logger
import argparse
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os
import pandas as pd
from utils import get_dataset_and_dict

def img_generate():
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars, imagenet_100')
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    args.dataset_name = 'cifar100' # cub, cifar10, cifar100, imagenet_100, scars

    preprocess = transforms.Compose([transforms.Resize(size=224, max_size=None, antialias=None),
                                 transforms.CenterCrop(size=(224, 224)),
                                 transforms.ToTensor()
                               ])

    dataset,class_names_dict = get_dataset_and_dict(preprocess,args)
    
    train_loader = DataLoader(dataset, num_workers=0, batch_size=256, shuffle=False,sampler=None,drop_last=False)

    # print(f'class_names_dict:{class_names_dict}')
    print(f'len(class_names_dict.keys()):{len(class_names_dict.keys())}')
    txt_file_path = f'./txt/{args.dataset_name}_a_photo_of_label.txt'
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
    
    
    if args.dataset_name == 'cifar100':
        cifar100_coarse_label_map = {
        19: 11, 29: 15, 0: 4, 11: 14, 1: 1, 86: 5, 90: 18, 28: 3, 23: 10, 31: 11, 39: 5,
        96: 17, 82: 2, 17: 9, 71: 10, 8: 18, 97: 8, 80: 16, 74: 16, 59: 17, 70: 2, 87: 5,
        84: 6, 64: 12, 52: 17, 42: 8, 47: 17, 65: 16, 21: 11, 22: 5, 81: 19, 24: 7, 78: 15,
        45: 13, 49: 10, 56: 17, 76: 9, 89: 19, 73: 1, 14: 7, 9: 3, 6: 7, 20: 6, 98: 14,
        36: 16, 55: 0, 72: 0, 43: 8, 51: 4, 35: 14, 83: 4, 33: 10, 27: 15, 53: 4, 92: 2,
        50: 16, 15: 11, 18: 7, 46: 14, 75: 12, 38: 11, 66: 12, 77: 13, 69: 19, 95: 0, 99: 13,
        93: 15, 4: 0, 61: 3, 94: 6, 68: 9, 34: 12, 32: 1, 88: 8, 67: 1, 30: 0, 62: 2, 63: 12,
        40: 5, 26: 13, 48: 18, 79: 13, 85: 19, 54: 2, 44: 15, 7: 7, 12: 9, 2: 14, 41: 19,
        37: 9, 13: 18, 25: 6, 10: 3, 57: 4, 5: 6, 60: 10, 91: 1, 3: 8, 58: 18, 16: 3,
        }
        with open(txt_file_path,'w') as txt_file:
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                images, class_labels, uq_idxs = batch
                print(class_labels, uq_idxs)
                for i in range(len(class_labels)):
                    class_label = class_labels[i].item()
                    fine_class_name = class_names_dict['fine'][class_label]
                    coarse_label = cifar100_coarse_label_map.get(class_label)
                    coarse_class_name = class_names_dict['coarse'][coarse_label]
                    txt_file.write(f'a photo of a {fine_class_name}, which is a type of {coarse_class_name}.')
                    txt_file.write('\n')
        txt_file.close() 

    else:
        with open(txt_file_path,'w') as txt_file:
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                images, class_labels, uq_idxs = batch
                print(class_labels, uq_idxs)
                for i in range(len(class_labels)):
                    class_label = class_labels[i].item()
                    txt_file.write(f'a photo of {class_names_dict[class_label]}.')
                    txt_file.write('\n')
        txt_file.close()      

if __name__ == "__main__":
    img_generate()
    
    
