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
print("\n")

# downloading data
download_folder_contents('215050229179', os.path.join(ROOT_DIR, 'data'))

# 0.1 Folders and paths
# TODO:
import data.models.model8.model as m

model_folder = m.get_model_folder()
exp = 'exp_00'
cashe_fname = "D00"
tr_cashe_path = model_folder + r'tr_features_' + cashe_fname + '.pkl'
val_cashe_path = model_folder + r'val_features_' + cashe_fname + '.pkl'

weight_path = model_folder + r'checkpoints/' + exp
metrics_path = weight_path + "_pkl.pickle"

# 1. Loading Data


batch_size = 5000
eval_epoch = 5  # evaluation every # epochs

net = m.mh_Net()
num_epochs = 20
learning_rate = 0.001

# 1.3 sampling and converting to dataloaders

ignore_pickable=True
tr_dataset = m.pc_labeled_dataset(tr_cashe_path, r'data/pc_labeled/train/', 20, ignore_pickable=ignore_pickable)
val_dataset = m.pc_labeled_dataset(val_cashe_path, r'data/New folder/val', 20, ignore_pickable=ignore_pickable)
ts_dataset = m.pc_labeled_dataset(val_cashe_path, r'data/New folder/train/', 20, ignore_pickable=ignore_pickable)

loader_tr = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
loader_ts = DataLoader(ts_dataset, batch_size=batch_size, shuffle=False)

# 2. Network
# 2.1 Setup and Architecture
n_classes = len(torch.unique(tr_dataset.y))
model_loss = m.my_loss(tr_dataset.y)  # the labels tensor
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
summary(net, tr_dataset.features[0].shape)

# 2.2 Train
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Training phase
    sTime1 = time.time()
    net.train()
    epoch_loss = 0
    for batch, (inputs, targets) in enumerate(loader_tr):
        inputs, targets = inputs.to(device), targets.squeeze().to(device)
        print(f"\r---[{batch + ((epoch - 1) % eval_epoch) * len(loader_tr) + 1}/{eval_epoch * len(loader_tr)}]---   ",
              end='', flush=True)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = model_loss.get_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss / len(loader_tr.dataset)

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
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.5f}, Validation Loss: {val_loss:.5f}"
              f"\t\tValidation acc: {val_ac:.5f}, mIoU: {miou}")
        print(iou)
        print(f"Elapsed time: {t1}")

        # Check if the current model has the best validation loss so far, and save it if it does
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), weight_path)

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
with torch.no_grad():
    for X, y in loader_ts:
        X, y = X.to(device), y.squeeze().to(device)
        output = torch.concat((output, net(X))) if output is not None else net(X)

y_pred = output.argmax(1).detach().cpu().numpy()
labels_ts = ts_dataset.y.cpu().numpy()
my_hist(y_pred)

plot_confusion_matrix_percent(labels_ts, y_pred, ['flat', 'edge', 'outlier', 'pickable'])

print("debug")
