#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Russell Izadi 2023
"""

from munch import Munch
import yaml
import os
import logging
from typing import Tuple
import math

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torchaudio

from vocalset import VOCALSET
import random
from itertools import accumulate
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import tree
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

import torch
from torch.utils.data import Dataset


class Transform(Dataset):
    def __init__(self, dataset, transform):
        # Store the original dataset, transform function
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        # Return the length of the original dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Return the transformed sample
        return self.transform(self.dataset[idx])


class Pair(Dataset):
    """Pairs a sample with a random sample from the same class
    """

    def __init__(self, dataset, attr):
        self.dataset = dataset
        self.attr = attr

    def __getitem__(self, idx):
        path_wav, label = self.dataset[idx]
        indices = self.dataset.indices[self.attr][label[self.attr]]
        idx_ = random.choice(indices)
        path_wav_, label_ = self.dataset[idx_]
        assert label[self.attr] == label_[self.attr]
        return path_wav, path_wav_, self.attr

    def __len__(self):
        return len(self.dataset)


def read_yml(path):
    """Converts a YAML file into an object with hierarchical attributes

    Args:
        path (string): The path to the YAML file (.yml)

    Returns:
        args (Munch): A Munch instance
    """

    assert path.endswith(".yml")
    assert os.path.exists(path), path

    with open(path, 'r', encoding='ASCII') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    args = Munch().fromDict(args)
    return args


def str_to_int(string_list):
    unique_strings = list(set(string_list))
    string_to_int = {string: i for i, string in enumerate(unique_strings)}
    int_list = [string_to_int[string] for string in string_list]
    return int_list


def precision_at_k(distance_matrix, labels, k):
    N = distance_matrix.shape[0]  # Number of samples
    precisions = []

    # For each sample
    for i in range(N):
        # Get indices of the k nearest neighbors (excluding the sample itself)
        nearest_neighbors = np.argsort(distance_matrix[i])[1:k+1]
        # Get labels of the k nearest neighbors
        neighbor_labels = labels[nearest_neighbors]
        # Calculate precision at k for the sample
        precision = np.sum(labels[i] == neighbor_labels) / k
        precisions.append(precision)

    # Return mean precision at k
    return np.mean(precisions)


def mean_reciprocal_rank(distance_matrix, labels):
    N = distance_matrix.shape[0]  # Number of samples
    ranks = []

    # For each sample
    for i in range(N):
        # Get indices of the neighbors sorted by distance (excluding the sample itself)
        sorted_neighbors = np.argsort(distance_matrix[i])[1:]
        # Find the ranks of neighbors with the same label
        same_label_ranks = np.where(
            labels[sorted_neighbors] == labels[i])[0] + 1
        if len(same_label_ranks) > 0:
            # The reciprocal rank is the reciprocal of the rank of the first neighbor with the same label
            ranks.append(1.0 / same_label_ranks[0])

    # Return mean reciprocal rank
    return np.mean(ranks)


def mean_average_precision(distance_matrix, labels):
    N = distance_matrix.shape[0]  # Number of samples
    average_precisions = []

    # For each sample
    for i in range(N):
        # Get indices of the neighbors sorted by distance (excluding the sample itself)
        sorted_neighbors = np.argsort(distance_matrix[i])[1:]
        # Get labels of the sorted neighbors
        sorted_neighbor_labels = labels[sorted_neighbors]
        # Indicator function where each value is 1 if the corresponding neighbor's label matches the sample's label, 0 otherwise
        indicator = (sorted_neighbor_labels == labels[i])

        # Average Precision calculation
        precisions = np.cumsum(indicator) / \
            (np.arange(len(sorted_neighbor_labels)) + 1)
        if np.sum(indicator) > 0:
            average_precisions.append(
                np.sum(precisions * indicator) / np.sum(indicator))

    # Return mean of Average Precision
    return np.mean(average_precisions)


def ndcg(distance_matrix, labels):
    N = distance_matrix.shape[0]  # Number of samples
    ndcgs = []

    # For each sample
    for i in range(N):
        # Get indices of the neighbors sorted by distance (excluding the sample itself)
        sorted_neighbors = np.argsort(distance_matrix[i])[1:]
        # Get labels of the sorted neighbors
        sorted_neighbor_labels = labels[sorted_neighbors]

        # Relevance scores (1 if label is the same, 0 otherwise)
        relevance = (sorted_neighbor_labels == labels[i]).astype(float)

        # Discounted Cumulative Gain (DCG)
        gains = 2 ** relevance - 1
        discounts = np.log2(np.arange(2, gains.size + 2))
        dcg = np.sum(gains / discounts)

        # Ideal Discounted Cumulative Gain (IDCG)
        ideal_gains = np.sort(gains)[::-1]
        idcg = np.sum(ideal_gains / discounts)

        # Normalized Discounted Cumulative Gain (NDCG)
        if idcg > 0:
            ndcgs.append(dcg / idcg)

    # Return mean NDCG
    return np.mean(ndcgs)


def dataset_mean_std(dataset):

    mean_list = []
    std_list = []

    # Iterate over the dataset in batches
    for item in dataset:
        sample = item[0]
        mean_list.append(torch.mean(sample))
        std_list.append(torch.std(sample))

    # Calculate overall mean and standard deviation
    overall_mean = torch.mean(torch.stack(mean_list))
    overall_std = torch.stack(std_list).pow(2).mean().sqrt()

    return overall_mean, overall_std


def num_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
