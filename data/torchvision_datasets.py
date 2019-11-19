import torch
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
from models import model_attributes
from data.utils import *

### CIFAR10 ###
def load_CIFAR10(args, train):
    transform = get_transform_CIFAR10(args, train)
    dataset = torchvision.datasets.CIFAR10(args.root_dir, train, transform=transform, download=True)
    if train:
        subsets = train_val_split(dataset, args.val_fraction)
    else:
        subsets = [dataset,]
    return subsets

def get_transform_CIFAR10(args, train):
    transform_list = []
    # resize if needed
    target_resolution = model_attributes[args.model]['target_resolution']
    if target_resolution is not None:
        transform_list.append(transforms.Resize(target_resolution))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    composed_transform = transforms.Compose(transform_list)
    return composed_transform
