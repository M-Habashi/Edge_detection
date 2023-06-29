# 0.0 Imports
from utils.my_utils2 import *
from utils.visualize_utils import *

import os
import pickle
import time
from datetime import timedelta
from tqdm import tqdm

import torch
import torch.nn.functional as fun
import torch.nn as nn
from torchsummary import summary
import matplotlib as mlp
import importlib

mlp.use('Qt5Agg')
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n")

# 0.1 Folders and paths
import models.model7.model as m
model_folder = m.get_model_folder()
exp = 'exp_05'

weight_path = model_folder + r'checkpoints/' + exp
metrics_path = weight_path + "_pkl.pickle"

# 1. Loading Data
classes = {'InnerEdge': 1, 'Outlier': 2}

pc_files = get_files_in_folder(r'data/pc_labeled/test', 'pcd', ROOT_DIR)
pc_dict = {i: load_pcd(pc_files[i]) for i in range(len(pc_files))}

labels_files = get_files_in_folder(r'data/pc_labeled/test', 'json', ROOT_DIR)
labels_dict = {i: load_labels(labels_files[i], classes, pc_dict[i].shape[0]) for i in range(len(labels_files))}

# 2. Loading model and inference
print("Loading mh_Net")
net = m.mh_Net()
net.load_state_dict(torch.load(weight_path))
net.to(device)
net.eval()

#TODO:
for k in tqdm(range(250, 290)):
    if k % 18 != 0:
        continue
    pc = torch.tensor(pc_dict[k]).to(device)
    labels = torch.tensor(labels_dict[k]).to(device) if k < len(labels_dict) else torch.zeros((pc.shape[0])).to(device)
    title = pc_files[k].split("\\")[-1].split('.')[0]

    # 2. Inference
    print("Testing mh_Net")
    pc = m.pc_normalize(pc)
    # output_logts = m.mh_model(exp, pc, title)
    output_logts = m.mh_model(exp, pc)
    pred = output_logts.argmax(1)


    # softing the results (0 and value instead of 0 and 1)
    edge_pred_soft = torch.zeros(pred.shape).to(torch.float32).to(output_logts.device)

    edge_idx = torch.where((pred == 1) | (pred == 3))[0]
    #edge_prop_normalized = m.m_norm(output_logts[edge_idx, 1])
    edge_prop_normalized = output_logts[edge_idx, 1]
    edge_pred_soft[edge_idx] = edge_prop_normalized - torch.min(edge_prop_normalized)  # substracting the min to prevent ant negatice values

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    legends = ['inner', 'edges', 'outlier', 'pickable']
    colors = ['black', '#0773a6', '#b00505', '#51a607']
    visualize_labels(pc.cpu().numpy(), pred.cpu().numpy(),
                     v_colors=colors, fig=fig, ax=ax1, title=f"Original: {len(pred)} points", legends=legends)

    visualize_labels(pc[edge_idx].cpu().numpy(), pred[edge_idx].cpu().numpy(),
                     fig=fig, ax=ax2, title=f"Sampled points: {len(edge_idx)} points", legends=legends, v_colors=colors,
                     v_labels=pred.cpu().numpy())

    fig.suptitle(f"{k}: {title}")
    # visualize_values(pc.cpu().numpy(), edge_pred_soft.cpu().numpy(),
    #                   title="Edge prop", vmax=None)


    plt.show(block=False)
print("debug")

