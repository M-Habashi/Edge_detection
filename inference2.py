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
classes = {'InnerEdge': 1, 'Outlier': 2, 'Pickable': 3}
folder = r'data/val/'
k = 0

import data.models.model11BO.model as m
model_folder = m.get_model_folder()

# 1. Loading Data

pc_files = get_files_in_folder(folder, 'pcd', ROOT_DIR)
labels_files = get_files_in_folder(folder, 'json', ROOT_DIR)
labels_titles = [labels_files[i].split('/')[-1].split(".")[0] for i in range(len(labels_files))]

pc_dict, labels_dict = {}, {}
for i, pc_file in enumerate(pc_files):
    title = pc_files[i].split('/')[-1].split(".")[0]
    pc_dict[title] = load_pcd(pc_files[i])
    labels_dict[title] = load_labels(labels_files[labels_titles.index(title)], classes, pc_dict[title].shape[0]) if title in labels_titles \
        else torch.zeros(pc_dict[title].shape[0])

# 2. Loading model and inference

for k in tqdm(range(len(pc_files)-7)):
    title = pc_files[k].split("/")[-1].split('.')[0]
    pc = torch.tensor(pc_dict[title]).to(device)
    labels = m.adjust_labels_no_pickable(torch.tensor(labels_dict[title]).to(device))

    # 2. Inference
    print("Testing mh_Net")
    pred, idx = m.mh_model_pc('exp_06', pc)
    # pred, idx = m.mh_model_pc_hirerchial('exp_02','exp_edge_00', pc)
    pc = pc[idx]
    labels = labels[idx]

    acc = torch.mean((pred == labels).to(torch.float32))
    iou = m.compute_miou(labels.to(torch.int64), pred.to(torch.int64), 3)
    print(f"{title} : {iou}")

    edge_idx = torch.where((pred == 1) | (pred == 3))[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    legends = ['inner', 'edges', 'outlier', 'pickable']
    colors = ['black', '#0773a6', '#b00505', '#51a607']
    visualize_labels(pc.cpu().numpy(), labels.cpu().numpy(),
                     v_colors=colors, fig=fig, ax=ax1, title=f"Ground Truth: {len(pred)} points", legends=legends)

    visualize_labels(pc.cpu().numpy(), pred.cpu().numpy(),
                     fig=fig, ax=ax2, title=f"Prediction acc = {acc*100 :0.2f}%", legends=legends, v_colors=colors,
                     v_labels=pred.cpu().numpy())

    fig.suptitle(f"{k}: {title}")


plt.show(block=False)
print("debug")

