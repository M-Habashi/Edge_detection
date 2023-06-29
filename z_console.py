from utils.my_utils2 import *
from utils.visualize_utils import *
from utils.Knearest_utils import *
import os
import pickle
import time
from datetime import timedelta
import torch
import torch.nn.functional as fun
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import matplotlib as mlp

mlp.use('Qt5Agg')

import torch


def get_even_spaced_indices(arr, n):
    idx = torch.round(torch.linspace(0, len(arr)-1, n)).to(torch.int64)  # Generating sampled indices
    return idx.cpu()


# Example usage
rand_arr = torch.randn(200)
for i in tqdm(range(1, 101)):
    output_tensor = get_even_indices(rand_arr, i)
    if i!=len(output_tensor):
        print(f"i={i}\tout={len(output_tensor)}")

print('debug')
