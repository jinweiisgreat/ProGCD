from custom_dataset import *
import numpy as np
from torchvision.datasets.folder import default_loader
import os
import json

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

def get_dataset_and_dict(preprocess,args):   
   
    if args.dataset_name == "cifar10":
        dataset = CustomCIFAR10("/home/cifar10/", transform=preprocess, train=True)
        class_names_cifar10 = np.load("/home/cifar10/cifar-10-batches-py/batches.meta",allow_pickle=True) 
        class_names_arr = class_names_cifar10["label_names"]

        class_names_dict = {}
        for cid,class_name in enumerate(class_names_arr):
            class_names_dict[cid] = class_name
        
    elif args.dataset_name == "cifar100":
        dataset = CustomCIFAR100("/home/cifar100/", transform=preprocess, train=True)
        class_names_cifar100 = np.load("/home/cifar100/cifar-100-python/meta", allow_pickle=True) 
        fine_label_names = class_names_cifar100["fine_label_names"] 
        coarse_label_names = class_names_cifar100["coarse_label_names"]      

        class_names_dict = {"fine":{}, "coarse":{}}
        for cid,fine_class_name in enumerate(fine_label_names):
            class_names_dict["fine"][cid] = fine_class_name
        for cid,coarse_class_name in enumerate(coarse_label_names):
            class_names_dict["coarse"][cid] = coarse_class_name

        print(f'class_names_dict:{class_names_dict}')
            
    elif args.dataset_name == "imagenet_100":
        imagenet_root = '/home/ImageNet100/ILSVRC12'
        dataset = ImageNetBase(root=os.path.join(imagenet_root, 'train'), transform=preprocess)
        class_to_idx_map = dataset.class_to_idx
        labels_path = os.path.join(imagenet_root, 'Labels.json')
        with open(labels_path, 'r') as file:
            class_names_imagenet_dict = json.load(file)

        file.close()

        class_names_dict = {}
        for class_order in class_names_imagenet_dict.keys():
            real_class_name = class_names_imagenet_dict[class_order]
            class_id = class_to_idx_map[class_order]
            class_names_dict[class_id] = real_class_name   
    
    elif args.dataset_name == "cub":
        dataset = CustomCub2011("/home/CUB/", transform=preprocess, train=True)
        class_names_dict = {}
        with open(f"/home/CUB/CUB_200_2011/classes.txt",'r') as txt_file:
            for i,txt in enumerate(txt_file):
                txt = txt.replace("\n","")
                class_name = txt[txt.find(".")+1:]
                class_names_dict[i] = class_name

        txt_file.close()
        
    elif args.dataset_name == "aircraft":
        dataset = FGVCAircraft("/home/Aircraft/fgvc-aircraft-2013b/", transform=preprocess,split="trainval")
        class_names_dict = {}
        with open(f"/home/Aircraft/fgvc-aircraft-2013b/data/variants.txt",'r') as txt_file:
            for i,txt in enumerate(txt_file):
                txt = txt.replace("\n","")
                class_name = txt
                class_names_dict[i] = class_name
 
        txt_file.close()
    
    elif args.dataset_name == "scars":
        dataset = CarsDataset(transform=preprocess, train=True)
        class_names_dict = {}
        with open(f"/home/Stanford_cars/label_map.txt",'r') as txt_file:
            for i,txt in enumerate(txt_file):
                txt = txt.replace("\n","")
                class_name = txt
                class_names_dict[i] = class_name

        txt_file.close()
            
    return dataset,class_names_dict