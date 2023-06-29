import numpy as np
import torch
from sklearn import linear_model, datasets
from tqdm import tqdm


def get_prop_by_ratio_of_classes(pc, labels, k):
    def prop(x):
        return (x * (1 - x) * 4) ** 0.2

    parts_ind = [i for i in range(len(labels)) if labels[i] != 0]
    # try "%timeit [i for i in range(len(labels)) if labels[i] != 0]" in command line

    props = torch.zeros_like(labels, dtype=torch.float).to(pc.device)
    for i in parts_ind:
        distance = np.linalg.norm(pc - pc[i], axis=1)
        nearest_idx = np.argsort(distance)[:k]
        r = (labels[nearest_idx] == labels[i]).sum().item() / k
        props[i] = prop(r)
    return props


def get_outliers_ratio(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
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
    vals = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
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
    vals = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
    votes = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
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
    vals = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        d_mink, nearest_idx = torch.topk(distance, k=k, largest=False)
        vals[i] = torch.mean(d_mink)
    return vals


# the (di - dmean)^3 of the NKP (nearest k points) (like skewness coeff)
def get_dist_skewness(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        d_mink, nearest_idx = torch.topk(distance, k=k, largest=False)
        d_mean = torch.mean(d_mink)
        d_std = torch.std(d_mink)
        vals[i] = torch.abs(torch.sum((d_mink - d_mean) ** 3)) / (k * d_std ** 3)
    return vals


# the dist to the centroid of NKP (found to be not effective (Maybe with RANSAC))
def get_dist_to_cg_prop(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        _, nearest_idx = torch.topk(distance, k=k, largest=False)
        vals[i] = torch.norm(torch.mean(pc[nearest_idx], dim=0) - pc[i])  # the dist to cg
    return vals


def furthest_from_nearest_k(pc, k):
    vals = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        distance, nearest_idx = torch.topk(distance, k=k, largest=False)
        distance, nearest_idx = torch.topk(distance, k=1, largest=True)

        vals[i] = distance
    return vals


def furthest_multiplication_k1_k2(pc, k1, k2):
    vals = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        distance, nearest_idx = torch.topk(distance, k=k1, largest=False)

        vals[i] = distance[k1 - 1] * distance[k2 - 1]
    return vals


def furthest_addition_k1_k2(pc, k1, k2):
    vals = torch.zeros(pc.shape[0], dtype=torch.float).to(pc.device)
    for i in tqdm(range(len(vals))):
        distance = torch.norm(pc - pc[i], dim=1)
        distance, nearest_idx = torch.topk(distance, k=k1, largest=False)

        vals[i] = distance[k1 - 1] + distance[k2 - 1]
    return vals


def sample_point_cloud(pc, labels, n_output, k_ratio=0.01, outlier_threshold=2):
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


def mh_normalize(x):
    z = (x - torch.mean(x).item()) / torch.std(x).item()
    return z, torch.mean(x).item(), torch.std(x).item()


def point_features_old(pc):
    # input torch tensor n x 3
    # output torch tensor n x p
    n_points = pc.shape[0]
    k1 = int(0.1 * n_points)
    k2 = 20

    # Features
    f_pc = torch.zeros((n_points, 0)).to(pc.device)
    f_pc = torch.hstack((f_pc, furthest_from_nearest_k(pc, k1).unsqueeze(1)))
    f_pc = torch.hstack((f_pc, furthest_from_nearest_k(pc, k2).unsqueeze(1)))
    f_pc = torch.hstack((f_pc, get_dist_to_cg_prop(pc, k1).unsqueeze(1)))
    f_pc = torch.hstack((f_pc, get_dist_skewness(pc, k1).unsqueeze(1)))

    return f_pc
