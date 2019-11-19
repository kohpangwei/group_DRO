import torch
import numpy as np
from torch.utils.data import Subset

# Train val split
def train_val_split(dataset, val_frac):
    # split into train and val
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    val_size = int(np.round(len(dataset)*val_frac))
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    train_data, val_data = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_data, val_data

# Subsample a fraction for smaller training data
def subsample(dataset, fraction):
    indices = np.arange(len(dataset))
    num_to_retain = int(np.round(float(len(dataset)) * fraction))
    np.random.shuffle(indices)
    return Subset(dataset, indices[:num_to_retain])
