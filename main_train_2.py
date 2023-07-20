# 0.0 Imports
from utils.my_utils2 import *
from utils.visualize_utils import *
from utils.Knearest_utils import *
from settings import *
from utils.data_utils import download_folder_contents

import os
import pickle
import time
from datetime import timedelta

import torch
import torch.nn.functional as fun
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchsummary import summary
import matplotlib as mlp
mlp.use('Qt5Agg')
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("\n")

# downloading data
download_folder_contents('215050229179', os.path.join(ROOT_DIR, 'data'))

# 0.1 Folders and paths
# TODO:

# outliers model
import data.models.model11.model as m
exp_out = 'exp_02'
model_folder = m.get_model_folder()
weight_path = model_folder + r'checkpoints/' + exp_out
net_out = m.mh_Net_out()
net_out.load_state_dict(torch.load(weight_path))
net_out.to(device)
net_out.eval()


# EDge detection model
exp = 'exp_edge_00'
features_fname = "D00"
tr_cashe_path = model_folder + r'tr_features_' + features_fname + '.pkl'
val_cashe_path = model_folder + r'val_features_' + features_fname + '.pkl'

weight_path = model_folder + r'checkpoints/' + exp
metrics_path = weight_path + "_pkl.pickle"

# 1. Loading Data

batch_size = 1  # right noe can't be changed
eval_epoch = 5  # evaluation every # epochs

net = m.mh_Net_Edge()
num_epochs = 1
learning_rate = 0.0001

# 1.3 sampling and converting to dataloaders

ignore_pickable = True
tr_dataset = m.pc_labeled_dataset_by_folder(model_folder,  r'data/train', features_fname, rand_angle=20, n_rotations=2, sample_index=10, points_by_pc=8000)
val_dataset = m.pc_labeled_dataset_by_folder(model_folder, r'data/val', features_fname, points_by_pc=8000)
ts_dataset = m.pc_labeled_dataset_by_folder(model_folder, r'data/val', features_fname)

loader_tr = DataLoader(tr_dataset, batch_size=1, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
loader_ts = DataLoader(ts_dataset, batch_size=batch_size, shuffle=False)

# 2. Network
# 2.1 Setup and Architecture
n_classes = len(torch.unique(net.adjust_labels(tr_dataset.label_sample)))
model_loss = net.my_loss(tr_dataset.label_sample)  # the labels tensor
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train_losses, val_losses, train_acc, val_acc = \
    pickle.load(open(metrics_path, "rb")) \
        if os.path.exists(metrics_path) else ([], [], [], [])

os.makedirs(os.path.dirname(weight_path), exist_ok=True)
if os.path.exists(weight_path) and \
        m.check_state_dict_sizes(net.state_dict(), torch.load(weight_path)):

    print("Loading an existing model")
    net.load_state_dict(torch.load(weight_path))

else:
    print("Creating a new model")
    train_losses, val_losses, train_acc, val_acc = ([], [], [], [])

net.to(device)
summary(net, tr_dataset[0][0][0].shape)

# 2.2 Train
best_metric = - float('inf')
for epoch in range(num_epochs):
    # Training phase
    sTime1 = time.time()
    net.train()
    epoch_loss = 0
    n_losses = 0
    for batch, (inputs, targets, pc) in enumerate(loader_tr):
        inputs, targets, pc = inputs.to(device), targets.squeeze().to(device), pc.squeeze().to(device)
        inputs = inputs.squeeze(0)  #special input squeeze as it is batched already
        print(f"\r---[{batch + ((epoch - 1) % eval_epoch) * len(loader_tr) + 1}/{eval_epoch * len(loader_tr)}]---   ",
              end='', flush=True)

        # Remove outliers
        output_logts = net_out(inputs)
        pred_inlier_idx = torch.where(output_logts.argmax(1) == 0)[0]
        pc, inputs, targets = pc[pred_inlier_idx], inputs[pred_inlier_idx], targets[pred_inlier_idx]
        targets = net.adjust_labels(targets)
        # visualize_labels(pc, targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = model_loss.get_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        n_losses += len(targets)
        epoch_loss += loss

    epoch_loss = epoch_loss / n_losses
    # Evaluate
    if epoch % eval_epoch == 0:  # evaluating every 5 epochs increases the speed
        train_losses.append(epoch_loss.item())
        # Validation phase
        net.eval()
        val_loss, val_ac, miou, iou = m.evaluate_model(net, model_loss, loader_val, n_classes)
        val_losses.append(val_loss)
        val_acc.append(val_ac)

        with open(metrics_path, "wb") as f:
            pickle.dump((train_losses, val_losses, train_acc, val_acc), f)

        # Print the training and validation losses for the epoch
        t1 = str(timedelta(seconds=time.time() - sTime1))
        print("")
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss*10000:.5f}, Validation Loss: {val_loss*10000:.5f}"
              f"\tValidation acc: {val_ac:.5f}, mIoU: {miou}")
        print(iou)
        print(f"Elapsed time: {t1}")

        # Check if the current model has the best validation loss so far, and save it if it does
        metric = - val_loss
        if metric > best_metric:
            best_metric = metric
            torch.save(net.state_dict(), weight_path)
            print("***Saving Model***")

# 3. Plotting Metrics
plt.figure()
plt.plot(train_losses, label='Training Loss', color='black')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.show(block=False)

print('------------------------------------------')
test_loss, test_ac, miou_test, _ = m.evaluate_model(net, model_loss, loader_ts, n_classes)
print(f"Testing acc: {test_ac:.5f}, Testing loss: {test_loss:.5f},  mIoU={miou_test}")

plt.show(block=False)
print('Finished Training')

# confusion matrix for testing
output = None
target = None
with torch.no_grad():
    for X, y, pc in loader_ts:
        X, y = X.to(device), y.squeeze().to(device)
        X = X.squeeze(0)  # special input squeeze as it is batched already
        y = net.adjust_labels(y)

        output = torch.concat((output, net(X))) if output is not None else net(X)
        target = torch.concat((target, y)) if target is not None else y

y_pred = output.argmax(1).detach().cpu().numpy()
y_true = target.cpu().numpy()
my_hist(y_pred)

plot_confusion_matrix_percent(y_true, y_pred, ['flat', 'edge'])

print("debug")
