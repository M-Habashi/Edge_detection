import os
import numpy as np
import torch


def float_list(my_list):
    my_float_list = []
    for x in my_list:
        try:
            my_float_list.append(float(x))
        except ValueError:
            my_float_list.append(x)

    return my_float_list


def uniform_sample(features, labels, icol):
    # icol is the column to be uniform (starting from zero index)
    # for uniform labels let icol = n_features
    t = torch.hstack((features, labels.squeeze().unsqueeze(1)))
    uniform_t = torch.zeros((0, t.shape[1])).to(features.device)

    n = min([len(t[t[:, icol] == i]) for i in torch.unique(labels)])
    for i in torch.unique(t[:, icol]).int().tolist():
        tensor_i = t[t[:, icol] == i]
        uniform_t = torch.vstack((uniform_t, tensor_i[:n]))

    n_features = features.shape[1]
    input_u, labels_u = uniform_t[:, :n_features], uniform_t[:, n_features:]

    return input_u, labels_u


def uniform_sample2(features, labels, icol, jcol, ratio_tr, ratio_val):
    t = torch.hstack((features, labels))
    data_tr = torch.zeros((0, t.shape[1]))
    data_val = torch.zeros((0, t.shape[1]))
    data_ts = torch.zeros((0, t.shape[1]))

    for i in torch.unique(t[:, icol]).int().tolist():
        tensor_i = t[t[:, icol] == i]

        for j in torch.unique(tensor_i[:, jcol]).int().tolist():
            tensor_j = tensor_i[tensor_i[:, jcol] == j]
            add_tr, add_val, add_ts = mh_divide(tensor_j, ratio_tr, ratio_val)

            data_tr = torch.vstack((data_tr, add_tr[:2000]))
            data_val = torch.vstack((data_val, add_val[:2000]))
            data_ts = torch.vstack((data_ts, add_ts[:2000]))

    n_features = features.shape[1]
    input_tr, labels_tr = data_tr[:, :n_features], data_tr[:, n_features:]
    input_val, labels_val = data_val[:, :n_features], data_val[:, n_features:]
    input_ts, labels_ts = data_ts[:, :n_features], data_ts[:, n_features:]

    return input_tr, labels_tr.to(torch.long), input_val, labels_val.to(torch.long), input_ts, labels_ts.to(torch.long)


def mh_divide(inputs, labels, r_tr, r_val):

    N = inputs.shape[0]
    ind = torch.randperm(N)

    n_tr = int(N * r_tr)
    n_val = int(N * r_val)
    n_ts = N - n_tr - n_val

    input_tr, labels_tr = inputs[ind[:n_tr]], labels[ind[:n_tr]]
    input_val, labels_val = inputs[ind[n_tr:n_tr + n_val]], labels[ind[n_tr:n_tr + n_val]]
    input_ts, labels_ts = inputs[ind[n_tr + n_val:]], labels[ind[n_tr + n_val:]]

    return input_tr, labels_tr, input_val, labels_val, input_ts, labels_ts
