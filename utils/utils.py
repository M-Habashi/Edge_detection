import numpy as np
import torch
from sklearn import linear_model
from tqdm import tqdm


def shuffle_array(array):
    array = np.array(array)
    indices = np.random.permutation(len(array))
    shuffled_array = array[indices]
    return shuffled_array, indices


def stratified_sampling(arr, num):
    ratio = num / arr.shape[0]
    sample_ind = np.zeros(0)
    for c in np.unique(arr):
        c_ind, _ = shuffle_array(np.where(arr == c)[0])
        c_ind = c_ind[:int(len(c_ind) * ratio) + 1]
        sample_ind = np.append(sample_ind, c_ind)

    sample_ind = sample_ind[:num].astype(int)

    return sample_ind


def uniform_sampling(arr, num):
    unique_arr = np.unique(arr)

    # get an array with sorted unique classes based on frequency
    # important when there is a class that doesn't satisfy the required points (we should start with it)
    c_freq = {}
    for c in unique_arr:
        c_freq[c] = len(np.where(arr == c)[0])
    c_freq_sorted = dict(sorted(c_freq.items(), key=lambda x: x[1]))

    sample_ind = np.zeros(0)
    for i, c in enumerate(list(c_freq_sorted)):
        # get the number of points for each class based on the total remaining required points
        n_class = int((num - len(sample_ind)) / (len(unique_arr) - i))

        # get the indices of this class after shuffling the points
        c_ind, _ = shuffle_array(np.where(arr == c)[0])
        c_ind = c_ind[:n_class]

        # add to the global arr
        sample_ind = np.append(sample_ind, c_ind)

    sample_ind = shuffle_array(sample_ind)[0].astype(int)
    return sample_ind


def mh_divide(inputs, labels, r_tr, r_val):
    t = torch.hstack((inputs, labels.squeeze().unsqueeze(1)))
    t = t[torch.randperm(t.size(0))]

    N = t.shape[0]
    n_tr = int(N * r_tr)
    n_val = int(N * r_val)
    n_ts = N - n_tr - n_val

    t_tr = t[:n_tr, :]
    t_val = t[n_tr:n_tr + n_val, :]
    t_ts = t[n_tr + n_val:, :]

    n_features = inputs.shape[1]
    input_tr, labels_tr = t_tr[:, :n_features], t_tr[:, n_features:]
    input_val, labels_val = t_val[:, :n_features], t_val[:, n_features:]
    input_ts, labels_ts = t_ts[:, :n_features], t_ts[:, n_features:]

    return input_tr, labels_tr, input_val, labels_val, input_ts, labels_ts


def get_even_spaced_indices(arr, n):
    idx = torch.round(torch.linspace(0, len(arr) - 1, n)).to(torch.int64)  # Generating sampled indices
    return idx.cpu().numpy()


# ----------------------------------------------------------------------------------#
# delete or move
def get_prop_by_ratio_of_classes(pc, labels, k):
    def prop(x):
        return (x * (1 - x) * 4) ** 0.2

    parts_ind = [i for i in range(len(labels)) if labels[i] != 0]
    # try "%timeit [i for i in range(len(labels)) if labels[i] != 0]" in command line

    props = torch.zeros_like(labels, dtype=torch.float)
    for i in parts_ind:
        distance = np.linalg.norm(pc - pc[i], axis=1)
        nearest_idx = np.argsort(distance)[:k]
        r = (labels[nearest_idx] == labels[i]).sum().item() / k
        props[i] = prop(r)
    return props


def get_outliers_ratio(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float)
    for i in tqdm(range(len(vals))):
        distance = np.linalg.norm(pc - pc[i], axis=1)
        nearest_idx = np.argsort(distance)[:k]

        # performing ransac
        threshold = np.median(np.absolute(pc[nearest_idx, 2] - np.median(pc[nearest_idx, 2])))
        ransac = linear_model.RANSACRegressor(residual_threshold=threshold)
        ransac.fit(pc[nearest_idx, 0:2], pc[nearest_idx, 2])
        vals[i] = np.sum(ransac.inlier_mask_) / k

    print("vals ended")
    return vals


def get_if_outliers(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float)
    for i in tqdm(range(len(vals))):
        distance = np.linalg.norm(pc - pc[i], axis=1)
        nearest_idx = np.argsort(distance)[:k]

        # performing ransac
        ransac = linear_model.RANSACRegressor()
        ransac.fit(pc[nearest_idx, 0:2], pc[nearest_idx, 2])
        vals[i] = 1 if ransac.inlier_mask_[0] else 0

    print("vals ended")
    return vals


def RANSAC_voting(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float)
    votes = torch.zeros(pc.shape[0], dtype=torch.float)
    for i in tqdm(range(len(vals))):
        distance = np.linalg.norm(pc - pc[i], axis=1)
        nearest_idx = np.argsort(distance)[:k]

        # performing ransac
        ransac = linear_model.RANSACRegressor()
        ransac.fit(pc[nearest_idx, 0:2], pc[nearest_idx, 2])
        vals[nearest_idx] = vals[nearest_idx] + ransac.inlier_mask_

    return vals / vals.max()


# the mean distances of the NKP (nearest k points)
def get_dist_prop(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        d_mink, nearest_idx = torch.topk(distance, k=k, largest=False)
        vals[i] = torch.mean(d_mink)
    return vals


# the dist to the centroid of NKP (found to be not effective (Maybe with RANSAC))
def get_dist_to_cg_prop(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        _, nearest_idx = torch.topk(distance, k=k, largest=False)
        vals[i] = torch.norm(torch.mean(pc[nearest_idx], dim=0) - pc[i])  # the dist to cg
    return vals


def furthest_from_nearest_k(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        distance, nearest_idx = torch.topk(distance, k=k, largest=False)
        distance, nearest_idx = torch.topk(distance, k=1, largest=True)

        vals[i] = distance
    return vals


def furthest_multiplication_k1_k2(pc, k1, k2):
    vals = torch.zeros(pc.shape[0], dtype=torch.float)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        distance, nearest_idx = torch.topk(distance, k=k1, largest=False)

        vals[i] = distance[k1 - 1] * distance[k2 - 1]
    return vals


def sample_point_cloud(pc, labels, n_output, k_ratio=0.01, outlier_threshold=0.5):
    k = int(len(labels) * k_ratio)
    p_out = furthest_multiplication_k1_k2(pc, k, int(0.1 * k))
    p_norm = (p_out - torch.mean(p_out).item()) / torch.std(p_out).item()
    p_thr = torch.mean(p_out).item() + torch.std(p_out).item() * outlier_threshold
    inlier_idx = torch.where((p_norm <= p_thr))[0]
    pc = pc[inlier_idx, :]  # outliers removed
    labels = labels[inlier_idx]

    p_edge = get_dist_to_cg_prop(pc, k)
    _, idx_sampled = torch.topk(p_edge, k=n_output, largest=True)
    pc = pc[idx_sampled]
    labels = labels[idx_sampled]

    return pc, labels
