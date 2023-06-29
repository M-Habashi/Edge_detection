import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import os
import shutil
import json

from tabulate import tabulate
import humanize
import gc

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
import torch
import faiss

mlp.use('TkAgg')


# Files and loading

def create_directory(path):
    # Creates a directory and replaces it if existing
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)  # Removes all the subdirectories!
        os.makedirs(path)


def get_files_in_folder(folder_path, ext, ROOT_DIR):
    abs_folder_path = os.path.abspath(folder_path)
    file_list = []
    for root, dirs, files in os.walk(abs_folder_path):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, ROOT_DIR)
                file_list.append(rel_path)
    return file_list


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def load_pcd(pcd_file_path):
    array = []
    with open(pcd_file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line == 'DATA ascii':
                continue
            values = line.split()
            if len(values) == 3:
                xyz = [float(values[0]), float(values[1]), float(values[2])]
                array.append(xyz)
    return np.array(array)


def load_labels(label_file, classes, n_points):
    # Loads the labels from supervisely json files
    label_json = load_json(label_file)
    labels = np.zeros(n_points)
    indices = {key: [] for key in classes}
    for i_class, val in classes.items():
        for j_object in label_json['objects']:
            if j_object['classTitle'] != i_class:
                continue

            for k_figure in label_json['figures']:
                # could be id or key (don't know based on what)
                try:
                    if j_object['id'] != k_figure['objectId']:
                        continue
                except:
                    if j_object['key'] != k_figure['objectKey']:
                        continue

                indices[i_class].extend(k_figure['geometry']['indices'])

        labels[indices[i_class]] = val

    return labels


# Load pc and labels from text file
def import_pc_txt(file):
    data = np.loadtxt(file)
    loaded_pc = np.array(data[:, :3])
    loaded_ann = np.array(data[:, 3:]).squeeze()
    loaded_title = file.split('\\')[-1].split('.')[0]
    return loaded_pc, loaded_ann, loaded_title


# KNN algorithms

def show_nearest_neighbours(pc, i_point, k):
    k_label = np.zeros(pc.shape[0])
    distances = np.linalg.norm(pc - pc[i_point], axis=1)
    nearest_indices = np.argsort(distances)[:k]
    k_label[nearest_indices] = 1
    visualize_labels(pc, k_label, title=f"k={300}")
    return k_label

def show_nearest_neighbours_faiss(pc, i_point, k):

    k_label = np.zeros(pc.shape[0])

    faiss_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), pc.shape[1])
    faiss_index.add(pc.cpu().numpy())
    _, nearest_indices = faiss_index.search(pc[i_point].unsqueeze(0).cpu().numpy(), k=k)

    k_label[nearest_indices] = 1
    visualize_labels(pc, k_label, title=f"k={300}")
    return k_label

def visualize_labels(point_cloud, labels, title='Untitled', s=0.1, a=1, v_labels=None, v_colors=None, fig=None,
                     ax=None, legends=None):
    if type(point_cloud) == torch.Tensor:
        point_cloud = point_cloud.cpu().numpy()
    if type(labels) == torch.Tensor:
        labels = labels.cpu().numpy()

    point_cloud = point_cloud - np.mean(point_cloud, 0)
    if v_colors is None:
        v_colors = ['black', 'red', 'yellow', 'green', 'blue', 'pink']
    if v_labels is None:
        v_labels = np.unique(labels)
    else:
        v_labels = np.unique(v_labels)
    if legends is None:
        legends = np.unique(v_labels)

    df = pd.DataFrame({
        "x": point_cloud[:, 0],
        "y": point_cloud[:, 1],
        "z": point_cloud[:, 2],
        "label": labels
    })

    if fig == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.set_title(title)

    for label, color, legend in zip(v_labels, v_colors, legends):
        c_df = df[df["label"] == label]
        ax.scatter(c_df["x"], c_df["y"], c_df["z"], label=legend, alpha=a, c=color, s=s)

    ax.legend()
    legend = ax.legend()

    # Increase the size of the circles in the legend
    for handle in legend.legendHandles:
        handle.set_sizes([30])

    # make the scale of all axes the same
    ax.set_box_aspect([1, 1, 1])
    min_value = np.min(point_cloud)
    max_value = np.max(point_cloud)
    ax.set_xlim([min_value, max_value])
    ax.set_ylim([min_value, max_value])
    ax.set_zlim([min_value, max_value])

    plt.show(block=False)


def create_black_to_red_cmap():
    colors = [(0, 0, 0), (1, 0, 0)]  # Start with black and end with red
    cmap_name = 'BlackRed'
    return LinearSegmentedColormap.from_list(cmap_name, colors)


def create_black_to_blue_cmap():
    colors = [(0, 0, 0), (0, 0, 1)]  # Start with black and end with red
    cmap_name = 'BlackRed'
    return LinearSegmentedColormap.from_list(cmap_name, colors)


def create_red_to_blue_cmap():
    colors = [(1, 0, 0), (0, 0, 1)]  # Start with black and end with red
    cmap_name = 'RedBlue'
    return LinearSegmentedColormap.from_list(cmap_name, colors)


def visualize_values(point_cloud, values, title, c_map=None, fig=None, ax=None, vmin=None, vmax=None, s=0.1):
    point_cloud = point_cloud - np.mean(point_cloud, 0)
    if fig == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if c_map is None:
        c_map = create_black_to_blue_cmap()
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
               c=values, cmap=c_map, s=s, vmin=vmin, vmax=vmax)

    fig.colorbar(ax.collections[0])
    ax.set_title(title)
    # make the scale of all axes the same
    ax.set_box_aspect([1, 1, 1])
    min_value = np.min(point_cloud)
    max_value = np.max(point_cloud)
    ax.set_xlim([min_value, max_value])
    ax.set_ylim([min_value, max_value])
    ax.set_zlim([min_value, max_value])
    plt.show(block=False)


def my_hist(arr, n_pins=10):
    plt.figure()
    plt.hist(arr, bins=n_pins)
    plt.title("Histogram of Array")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show(block=False)


def plot_confusion_matrix(true_y, predicted_y, labels):
    plt.figure()
    # Create confusion matrix
    cm = confusion_matrix(true_y, predicted_y)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Add labels to each cell
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.show()


def plot_confusion_matrix_percent(true_y, predicted_y, labels):
    plt.figure()
    # Create confusion matrix
    cm = confusion_matrix(true_y, predicted_y)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Add labels to each cell
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f') + '%',
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.show()


class gpu_memory_usage:
    def __init__(self):
        self.table = []
        self.headers = ["#", "Allocated Memory", "Reserved Memory", "Free Memory"]

    def add(self, ind="NA"):
        allocated_memory = humanize.naturalsize(torch.cuda.memory_allocated())
        reserved_memory = humanize.naturalsize(torch.cuda.memory_reserved())
        free_memory = humanize.naturalsize(torch.cuda.memory_reserved() - torch.cuda.memory_allocated())

        row = [ind, allocated_memory, reserved_memory, free_memory]
        self.table.append(row)

    def print(self):
        table_str = tabulate(self.table, headers=self.headers, tablefmt="pretty")
        print(table_str)


def print_allocated_tensors():
    # Get the list of allocated tensors
    allocated_tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda]

    # Sort tensors by size in ascending order
    allocated_tensors = sorted(allocated_tensors, key=lambda tensor: tensor.element_size() * tensor.numel())

    # Prepare the table headers
    headers = ["Tensor Name", "Size", "Allocated Memory"]

    # Prepare the table rows
    table = []
    for i, tensor in enumerate(allocated_tensors):
        size = str(tensor.size())
        allocated_memory = humanize.naturalsize(tensor.element_size() * tensor.numel())
        table.append([f"Tensor {i + 1}", size, allocated_memory])

    # Print the table
    table_str = tabulate(table, headers=headers, tablefmt="orgtbl")
    print(table_str)


print()
