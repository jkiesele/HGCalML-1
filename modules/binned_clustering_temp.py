import gzip
import math
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from core.OCHits2Showers import OCHits2Showers
from numba import njit
from bin_by_coordinates_op import BinByCoordinates


@njit
def collect_func_native_gen(low_bin_indices,high_bin_indices, n_bins,flat_bin_finding_vector, assignment, coords, alpha_coords, alpha_radius,
                        shower_idx, beta, row_splits, beta_filtered, beta_filtered_indices_full):

    range_bin_indices = high_bin_indices - low_bin_indices
    ndim = len(high_bin_indices)

    bin_finding_div_vector = np.ones_like(low_bin_indices)
    for i in range(ndim - 1):
        bin_finding_div_vector[ndim - i - 2] = bin_finding_div_vector[ndim - i - 1] * range_bin_indices[ndim - i - 1]

    total_iterations = np.prod(range_bin_indices)

    for iteration in range(total_iterations):
        bin_vector = low_bin_indices + (iteration // bin_finding_div_vector) % range_bin_indices
        b_flat = np.sum(flat_bin_finding_vector * bin_vector)

        # print(bin_vector)

        if np.any(bin_vector >= n_bins):
            continue

        start_index = row_splits[b_flat]
        end_index = row_splits[b_flat + 1]
        for l in range(start_index, end_index):
            if assignment[l] == -1:
                if np.sum((coords[l] - alpha_coords) ** 2) < alpha_radius ** 2:
                    assignment[l] = shower_idx
                    beta[l] = 0
                    if beta_filtered_indices_full[l] != -1:
                        beta_filtered[beta_filtered_indices_full[l]] = 0

@njit
def collect_func_native(ix_l, ix_h, iy_l, iy_h, iz_l, iz_h, nbins, assignment, coords, alpha_coords, alpha_radius,
                        shower_idx, beta, row_splits, beta_filtered, beta_filtered_indices_full):
    x = 0
    for i in range(ix_l, ix_h):
        if i < 0 or i >= nbins:
            continue
        for j in range(iy_l, iy_h):
            if j < 0 or j >= nbins:
                continue
            for k in range(iz_l, iz_h):
                if k < 0 or k >= nbins:
                    continue
                b_flat = i * nbins ** 2 + j * nbins + k
                start_index = row_splits[b_flat]
                end_index = row_splits[b_flat + 1]
                for l in range(start_index, end_index):
                    if assignment[l] == -1:
                        if np.sum((coords[l] - alpha_coords) ** 2) < alpha_radius ** 2:
                            assignment[l] = shower_idx
                            beta[l] = 0
                            x += 1
                            if beta_filtered_indices_full[l] != -1:
                                beta_filtered[beta_filtered_indices_full[l]] = 0
    return x



def reconstruct_showers_no_op(coords, beta, beta_threshold=0.3, dist_threshold=1.5, pred_dist=None):
    beta = beta[:, 0]
    coords = coords - np.min(coords, axis=0, keepdims=True)
    pred_dist = pred_dist[:, 0] * dist_threshold

    _, bins_flat, n_bins, bin_width, _ = BinByCoordinates(coords, [0, len(coords)], n_bins=30)

    bins_flat = bins_flat.numpy()
    n_bins = n_bins.numpy()
    bin_width = float(bin_width[0])

    bin_width_x = bin_width_y = bin_width_z = bin_width

    sorting_indices = np.argsort(bins_flat)
    beta = beta[sorting_indices]
    coords = coords[sorting_indices]
    bins_flat = bins_flat[sorting_indices]
    pred_dist = pred_dist[sorting_indices]

    row_splits = tf.ragged.segment_ids_to_row_splits(bins_flat,num_segments=np.prod(n_bins)).numpy()

    flat_bin_finding_vector = np.concatenate((np.flip(np.cumprod(np.flip(n_bins)))[1:], [1]))

    assignment = np.zeros_like(beta, dtype=np.int32) -1

    shower_idx = 0
    beta_filtered_indices = np.argwhere(beta > beta_threshold)[:, 0]
    beta_filtered_indices_full = (assignment * 0 -1).astype(np.int32)
    beta_filtered_indices_full[beta_filtered_indices] = np.arange(len(beta_filtered_indices))
    beta_filtered = np.array(beta[beta_filtered_indices])

    bin_width = np.array([bin_width_x, bin_width_y, bin_width_z])
    pred_shower_alpha_idx = assignment * 0

    while True:
        alpha_idx = beta_filtered_indices[np.argmax(beta_filtered)]
        max_beta = beta[alpha_idx]
        # print("max beta", max_beta)
        if max_beta < beta_threshold:
            break
        alpha_coords = coords[alpha_idx]
        alpha_radius = pred_dist[alpha_idx]



        low_bin_indices = np.floor((alpha_coords - alpha_radius)/bin_width).astype(np.int32)
        high_bin_indices = np.ceil((alpha_coords + alpha_radius)/bin_width).astype(np.int32)
        collect_func_native_gen(low_bin_indices, high_bin_indices, n_bins, flat_bin_finding_vector, assignment, coords, alpha_coords,
                                alpha_radius, shower_idx, beta, row_splits, beta_filtered, beta_filtered_indices_full)


        beta[alpha_idx] = 0
        pred_shower_alpha_idx[shower_idx] = alpha_idx
        shower_idx += 1

    assignment_2 = assignment * 0
    assignment_2[sorting_indices] = assignment
    pred_shower_alpha_idx = pred_shower_alpha_idx[0:shower_idx]
    pred_shower_alpha_idx = np.array([sorting_indices[i] for i in pred_shower_alpha_idx])

    return assignment_2, pred_shower_alpha_idx[0:shower_idx]