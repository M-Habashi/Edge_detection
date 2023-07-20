# Imports
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
import data.models.model11BO.model as m

mlp.use('Qt5Agg')
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_folder = m.get_model_folder()
exp_num = 0
log_file_name = ""

features_fname = "D00"
tr_cashe_path = model_folder + r'tr_features_' + features_fname + '.pkl'
val_cashe_path = model_folder + r'val_features_' + features_fname + '.pkl'

# loading datasets
ignore_pickable = True
batch_size = 1  # right noe can't be changed

tr_dataset = m.pc_labeled_dataset_by_folder(model_folder, r'data/train', features_fname, rand_angle=0, n_rotations=1,
                                            sample_index=10, points_by_pc=8000)
val_dataset = m.pc_labeled_dataset_by_folder(model_folder, r'data/train', features_fname, points_by_pc=8000)
ts_dataset = m.pc_labeled_dataset_by_folder(model_folder, r'data/train', features_fname)

loader_tr = DataLoader(tr_dataset, batch_size=1, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=1, shuffle=True)
loader_ts = DataLoader(ts_dataset, batch_size=1, shuffle=False)


def logg(message):
    global log_file_name
    if log_file_name == "":
        return

    with open(log_file_name, "a+") as file:
        file.write(message + "\n")
        file.flush()


def write_tensor_to_txt_old(tensor):
    global log_file_name
    if not torch.is_tensor(tensor):
        logg("Input must be a Torch tensor.")

    with open(log_file_name, "a+") as file:
        for item in tensor.view(-1):
            file.write(str(item.item()) + "\n")
        file.flush()


def write_tensor_to_txt(tensor):
    global log_file_name
    if not torch.is_tensor(tensor):
        logg("Input must be a Torch tensor.")

    with open(log_file_name, "a+") as file:
        tensor_shape = tensor.size()
        if len(tensor_shape) > 2:
            logg("The input tensor must be 2-dimensional or lower.")
            return

        for row in tensor:
            for item in row:
                file.write(f"{item.item():<8}")  # You can adjust the formatting for wider matrices
            file.write("\n")
        file.flush()


def train_dynamic(loss_wt_1, loss_wt_2, gamma, lr):
    # parameters

    learning_rate = lr

    net = m.mh_Net_3_dynamic()
    num_epochs = 200
    eval_epoch = 5  # evaluation every # epochs
    global exp_num
    global log_file_name
    exp_num += 1
    exp = f"exp_{exp_num:02d}"
    weight_path = model_folder + r'checkpoints/' + exp
    metrics_path = weight_path + "_pkl.pickle"
    log_file_name = weight_path + "_log.txt"

    logg(f"Initializing... {exp}")
    logg(f"loss_wt_1: {loss_wt_1}, loss_wt_2: {loss_wt_2}, gamma: {gamma}, lr: {lr}")

    # Network and architecture
    n_classes = len(torch.unique(net.adjust_labels(tr_dataset.label_sample)))
    model_loss = net.my_loss(loss_wt_1, loss_wt_2, gamma, labels=tr_dataset.label_sample)  # the labels tensor
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_losses, val_losses, train_acc, val_acc = \
        pickle.load(open(metrics_path, "rb")) \
            if os.path.exists(metrics_path) else ([], [], [], [])

    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    if os.path.exists(weight_path) and \
            m.check_state_dict_sizes(net.state_dict(), torch.load(weight_path)):

        logg("Loading an existing model")
        net.load_state_dict(torch.load(weight_path))

    else:
        logg("Creating a new model")
        train_losses, val_losses, train_acc, val_acc = ([], [], [], [])

    net.to(device)

    # Train
    best_metric = - float('inf')
    for epoch in range(num_epochs):
        # Training phase
        sTime1 = time.time()
        net.train()
        epoch_loss = 0
        n_losses = 0
        for batch, (inputs, targets, pc) in enumerate(loader_tr):
            inputs, targets = inputs.to(device), targets.squeeze().to(device)
            inputs = inputs.squeeze(0)  # special input squeeze as it is batched already
            # logg(
            #     f"\r---[{batch + ((epoch - 1) % eval_epoch) * len(loader_tr) + 1}/{eval_epoch * len(loader_tr)}]---   ",
            #     )

            targets = net.adjust_labels(targets)
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
            val_loss, val_ac, miou, iou, _ = m.evaluate_model(net, model_loss, loader_val, n_classes)
            val_losses.append(val_loss)
            val_acc.append(val_ac)

            with open(metrics_path, "wb") as f:
                pickle.dump((train_losses, val_losses, train_acc, val_acc), f)

            # Print the training and validation losses for the epoch
            t1 = str(timedelta(seconds=time.time() - sTime1))
            logg("")
            logg(
                f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss * 10000:.5f}, Validation Loss: {val_loss * 10000:.5f}"
                f"\tValidation acc: {val_ac:.5f}, mIoU: {miou}")
            logg(f"log iou: {iou}")

            logg(f"Elapsed time: {t1}")

            # Check if the current model has the best validation loss so far, and save it if it does
            metric = miou
            if metric > best_metric:
                best_metric = metric
                torch.save(net.state_dict(), weight_path)
                logg("***Saving Model***")

    logg('------------------------------------------')
    net.load_state_dict(torch.load(weight_path))
    test_loss, test_ac, miou_test, iou_test, confusion_mat = m.evaluate_model(net, model_loss, loader_ts, n_classes)
    logg(f"Testing acc: {test_ac:.5f}, Testing loss: {test_loss:.5f},  mIoU={miou_test}")

    cm = confusion_mat / torch.sum(confusion_mat, dim=1, keepdim=True)
    obj_loss = 1 * torch.sum(torch.diagonal(cm)) - 1 * (cm[1][0] + cm[1][2]) \
               - 0.3 * (cm[0][1] + cm[2][1]) - 0.1 * (cm[2][0] + cm[0][2])

    write_tensor_to_txt(cm)
    logg(f"objective loss for BO: {obj_loss}")

    logg("finished")
    return obj_loss.cpu()
