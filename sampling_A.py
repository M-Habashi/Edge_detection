# Proposed sampling method analysis

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
import models.model5.model as m  #

exp = 'exp_00'  #

# 1. Loading Data
classes = {'pick': 1}

pc_files = get_files_in_folder(r'data/pickabple_points', 'pcd', ROOT_DIR)
pc_all = {i: load_pcd(pc_files[i]) for i in range(len(pc_files))}

labels_files = get_files_in_folder(r'data/pickabple_points', 'json', ROOT_DIR)
labels_all = {i: load_labels(labels_files[i], classes, pc_all[i].shape[0]) for i in range(len(labels_files))}

pc_i = 1
pc = torch.tensor(pc_all[pc_i]).to(device)
labels = torch.tensor(labels_all[pc_i]).to(device)
title = pc_files[pc_i].split("\\")[-1].split('.')[0]

# 2. Inference

output_logts = m.mh_model(exp, pc, title)  #the titl is used to import features from cashe directly
out_pred = output_logts.argmax(1)
edge_idx = torch.where(out_pred==1)[0]

covered = torch.sum(labels[edge_idx]) / torch.sum(labels)
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
visualize_labels(pc.cpu().numpy(), labels.cpu().numpy(), title=f"{title}: {torch.sum(labels):.0f} pickable points",
                 fig=fig, ax=ax1)
visualize_labels(pc[edge_idx].cpu().numpy(), labels[edge_idx].cpu().numpy(),
                 title=f"Covered {covered*100:.2f}%", fig=fig, ax=ax2)

n = 4000

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
visualize_labels(pc[edge_idx].cpu().numpy(), labels[edge_idx].cpu().numpy(),
                 title=f"n_points: {len(edge_idx)}, Covered {covered*100:.2f}%", fig=fig, ax=ax1)

idx_SA1 = m.sampling_randomly_from_edges(exp, pc, n, title)
covered = torch.sum(labels[idx_SA1]) / torch.sum(labels) * 100
visualize_labels(pc[idx_SA1].cpu().numpy(), labels[idx_SA1].cpu().numpy(),
                 fig=fig, ax=ax2, title=f"n_points: {n}, Covered {covered:.2f}%")

numbers = torch.linspace(start=1000, steps=10, end=40000).to(torch.int64).cpu().numpy()

means_chart = None
std_chart = None
for n in tqdm(numbers):
    covered_t = None
    for _ in range(30):
        idx_SA1 = m.sampling_randomly_from_edges(exp, pc, n, title)
        covered = torch.sum(labels[idx_SA1]) / torch.sum(labels) * 100
        covered_t = torch.cat((covered_t, covered.unsqueeze(0)),
                              dim=-1) if covered_t is not None else covered.unsqueeze(0)

    means_chart = torch.cat((means_chart, torch.mean(covered_t).unsqueeze(0))) if means_chart is not None \
        else torch.mean(covered_t).unsqueeze(0)
    std_chart = torch.cat((std_chart, torch.std(covered_t).unsqueeze(0))) if std_chart is not None \
        else torch.std(covered_t).unsqueeze(0)

# plt.style.use('bmh')
plt.style.use('seaborn-white')
cl1='#16807c'
fig, ax = plt.subplots()
# ax.scatter(numbers.cpu().numpy(), means_chart.cpu().numpy())
means_chart, std_chart = means_chart.cpu().numpy(), std_chart.cpu().numpy()
ax.plot(numbers, means_chart, color=cl1, linestyle='-', linewidth=1, label='Using Edge Detection')
ax.plot(numbers, means_chart+1.96*std_chart, color=cl1, linestyle='--', linewidth=0.5)
ax.plot(numbers, means_chart-1.96*std_chart, color=cl1, linestyle='--', linewidth=0.5)
ax.fill_between(numbers, means_chart+1.96*std_chart, means_chart-1.96*std_chart, color=cl1, alpha=0.1)
ax.set_xlabel('Number of points')
ax.set_ylabel('Covered Percentage of Pickable Points')
ax.set_title('Comparison of Sampling Methods for Obtaining Pickable Points (95% Confidence Interval)')

means_chart = None
std_chart = None
for n in tqdm(numbers):
    covered_t = None
    for _ in range(30):
        idx_RS = torch.randperm(len(labels))[:n]
        covered = torch.sum(labels[idx_RS]) / torch.sum(labels) * 100
        covered_t = torch.cat((covered_t, covered.unsqueeze(0)),
                              dim=-1) if covered_t is not None else covered.unsqueeze(0)

    means_chart = torch.cat((means_chart, torch.mean(covered_t).unsqueeze(0))) if means_chart is not None \
        else torch.mean(covered_t).unsqueeze(0)
    std_chart = torch.cat((std_chart, torch.std(covered_t).unsqueeze(0))) if std_chart is not None \
        else torch.std(covered_t).unsqueeze(0)

cl1='#801616'
means_chart, std_chart = means_chart.cpu().numpy(), std_chart.cpu().numpy()
ax.plot(numbers, means_chart, color=cl1, linestyle='-', linewidth=1, label='Random Sampling')
ax.plot(numbers, means_chart+1.96*std_chart, color=cl1, linestyle='--', linewidth=0.5)
ax.plot(numbers, means_chart-1.96*std_chart, color=cl1, linestyle='--', linewidth=0.5)
ax.fill_between(numbers, means_chart+1.96*std_chart, means_chart-1.96*std_chart, color=cl1, alpha=0.1)


# covered_t = None
# fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
# for row in axes:
#     for ax in row:
#         idx_SA1 = m.sampling_randomly_from_edges(exp, pc, n, title)
#         covered = torch.sum(labels[idx_SA1]) / torch.sum(labels) * 100
#         covered_t = torch.cat((covered_t, covered.unsqueeze(0)), dim=-1) if covered_t is not None else covered.unsqueeze(0)
#         visualize_labels(pc[idx_SA1].cpu().numpy(), labels[idx_SA1].cpu().numpy(),
#                          fig=fig, ax=ax, title=f"Covered {covered:.2f}%")
# fig.suptitle(f"SA1 sampling with {n} points (mean= {torch.mean(covered_t):0.2f}, std= {torch.std(covered_t):0.2f})")

ax.legend()
plt.show(block=False)
print('debug')
#


