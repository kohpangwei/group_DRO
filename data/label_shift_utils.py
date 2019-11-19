import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.dro_dataset import DRODataset
from data.utils import *
from data.torchvision_datasets import *


########################
### DATA PREPARATION ###
########################

def prepare_label_shift_data(args, train):
    settings = label_shift_settings[args.dataset]
    data = settings['load_fn'](args, train)
    n_classes = settings['n_classes']
    if train:
        train_data, val_data = data
        if args.fraction<1:
            train_data = subsample(train_data, args.fraction)
        train_data = apply_label_shift(train_data, n_classes, args.shift_type, args.minority_fraction, args.imbalance_ratio)
        data = [train_data, val_data]
    dro_data = [DRODataset(subset, process_item_fn=settings['process_fn'], n_groups=n_classes, 
                           n_classes=n_classes, group_str_fn=settings['group_str_fn']) \
                for subset in data]
    return dro_data

##############
### SHIFTS ###
##############

def apply_label_shift(dataset, n_classes, shift_type, minority_frac, imbalance_ratio):
    assert shift_type.startswith('label_shift')
    if shift_type=='label_shift_step':
        return step_shift(dataset, n_classes, minority_frac, imbalance_ratio) 

def step_shift(dataset, n_classes, minority_frac, imbalance_ratio):
    # get y info
    y_array = []
    for x,y in dataset:
        y_array.append(y)
    y_array = torch.LongTensor(y_array)
    y_counts = ((torch.arange(n_classes).unsqueeze(1)==y_array).sum(1)).float()
    # figure out sample size for each class
    is_major = (torch.arange(n_classes) < (1-minority_frac)*n_classes).float()
    major_count = int(torch.min(is_major*y_counts + (1-is_major)*y_counts*imbalance_ratio).item())
    minor_count = int(np.floor(major_count/imbalance_ratio))
    print(y_counts, major_count, minor_count)
    # subsample
    sampled_indices = []
    for y in np.arange(n_classes):
        indices,  = np.where(y_array==y)
        np.random.shuffle(indices)
        if is_major[y]:
            sample_size = major_count
        else:
            sample_size = minor_count
        sampled_indices.append(indices[:sample_size])
    sampled_indices = torch.from_numpy(np.concatenate(sampled_indices))
    return Subset(dataset, sampled_indices)

###################
### PROCESS FNS ###
###################

def xy_to_xyy(data):
    x,y = data
    return x,y,y

#####################
### GROUP STR FNS ###
#####################

def group_str_CIFAR10(group_idx):
    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return f'Y = {group_idx} ({classes[group_idx]})'

################
### SETTINGS ###
################

label_shift_settings = {
    'CIFAR10':{
        'load_fn': load_CIFAR10,
        'group_str_fn': group_str_CIFAR10,
        'process_fn': xy_to_xyy,
        'n_classes':10
    }
}

