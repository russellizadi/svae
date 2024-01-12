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


# coding=utf-8
# Copyright 2018 Ubisoft La Forge Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def dci(factors, codes, continuous_factors=True, model='lasso'):
    ''' DCI metrics from C. Eastwood and C. K. I. Williams,
        “A framework for the quantitative evaluation of disentangled representations,”
        in ICLR, 2018.

    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param model:                           model to use for score computation
                                            either lasso or random_forest
    '''
    # TODO: Support for discrete data
    assert (continuous_factors), f'Only continuous factors are supported'

    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]

    # normalize in [0, 1] all columns
    factors = minmax_scale(factors)
    codes = minmax_scale(codes)

    # compute entropy matrix and informativeness per factor
    e_matrix = np.zeros((nb_factors, nb_codes))
    informativeness = np.zeros((nb_factors, ))
    for f in range(nb_factors):
        if model == 'lasso':
            informativeness[f], weights = _fit_lasso(
                factors[:, f].reshape(-1, 1), codes)
            e_matrix[f, :] = weights
        elif model == 'random_forest':
            informativeness[f], weights = _fit_random_forest(
                factors[:, f].reshape(-1, 1), codes)
            e_matrix[f, :] = weights
        else:
            raise ValueError("Regressor must be lasso or random_forest")

    # compute disentanglement per code
    rho = np.zeros((nb_codes, ))
    disentanglement = np.zeros((nb_codes, ))
    for c in range(nb_codes):
        # get importance weight for code c
        rho[c] = np.sum(e_matrix[:, c])
        if rho[c] == 0:
            disentanglement[c] = 0
            break

        # transform weights in probabilities
        prob = e_matrix[:, c] / rho[c]

        # compute entropy for code c
        H = 0
        for p in prob:
            if p:
                H -= p * math.log(p, len(prob))

        # get disentanglement score
        disentanglement[c] = 1 - H

    # compute final disentanglement
    if np.sum(rho):
        rho = rho / np.sum(rho)
    else:
        rho = rho * 0

    # compute completeness
    completeness = np.zeros((nb_factors, ))
    for f in range(nb_factors):
        if np.sum(e_matrix[f, :]) != 0:
            prob = e_matrix[f, :] / np.sum(e_matrix[f, :])
        else:
            prob = np.ones((len(e_matrix[f, :]), 1)) / len(e_matrix[f, :])

        # compute entropy for code c
        H = 0
        for p in prob:
            if p:
                H -= p * math.log(p, len(prob))

        # get disentanglement score
        completeness[f] = 1 - H

    # average all results
    # disentanglement = np.dot(disentanglement, rho)
    # completeness = np.mean(completeness)
    # informativeness = np.mean(informativeness)

    disentanglement = disentanglement * rho
    completeness = completeness.reshape(-1, nb_factors//nb_codes)
    completeness = np.mean(completeness, axis=1)
    informativeness = informativeness.reshape(-1, nb_factors//nb_codes)
    informativeness = np.mean(informativeness, axis=1)
    return disentanglement, completeness, informativeness


def _fit_lasso(factors, codes):
    ''' Fit a Lasso regressor on the data

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    '''
    # alpha values to try
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]

    # make sure factors are N by 1
    factors.reshape(-1, 1)

    # find the optimal alpha regularization parameter
    best_a = 0
    best_mse = 10e10
    for a in alphas:
        # perform cross validation on the tree classifiers
        clf = Lasso(alpha=a, max_iter=5000)
        mse = cross_val_score(clf, codes, factors, cv=10,
                              scoring='neg_mean_squared_error')
        mse = -mse.mean()

        if mse < best_mse:
            best_mse = mse
            best_a = a

    # train the model using the best performing parameter
    clf = Lasso(alpha=best_a)
    clf.fit(codes, factors)

    # make predictions using the testing set
    y_pred = clf.predict(codes)

    # compute informativeness from prediction error (max value for mse/2 = 1/12)
    mse = mean_squared_error(y_pred, factors)
    informativeness = max(1 - 12 * mse, 0)

    # get the weight from the regressor
    predictor_weights = np.ravel(np.abs(clf.coef_))

    return informativeness, predictor_weights


def _fit_random_forest(factors, codes):
    ''' Fit a Random Forest regressor on the data

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    '''
    # alpha values to try
    max_depth = [8, 16, 32, 64, 128]
    max_features = [0.2, 0.4, 0.8, 1.0]

    # make sure factors are N by 0
    factors = np.ravel(factors)

    # find the optimal alpha regularization parameter
    best_mse = 10e10
    best_mf = 0
    best_md = 0
    for md in max_depth:
        for mf in max_features:
            # perform cross validation on the tree classifiers
            clf = RandomForestRegressor(
                n_estimators=10, max_depth=md, max_features=mf)
            mse = cross_val_score(clf, codes, factors,
                                  cv=10, scoring='neg_mean_squared_error')
            mse = -mse.mean()

            if mse < best_mse:
                best_mse = mse
                best_mf = mf
                best_md = md

    # train the model using the best performing parameter
    clf = RandomForestRegressor(
        n_estimators=10, max_depth=best_md, max_features=best_mf)
    clf.fit(codes, factors)

    # make predictions using the testing set
    y_pred = clf.predict(codes)

    # compute informativeness from prediction error (max value for mse/2 = 1/12)
    mse = mean_squared_error(y_pred, factors)
    informativeness = max(1 - 12 * mse, 0)

    # get the weight from the regressor
    predictor_weights = clf.feature_importances_

    return informativeness, predictor_weights


# coding=utf-8
# Copyright 2018 Ubisoft La Forge Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_bin_index(x, nb_bins):
    ''' Discretize input variable

    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization
    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)


def get_mutual_information(x, y, normalize=True):
    ''' Compute mutual information between two random variables

    :param x:      random variable
    :param y:      random variable
    '''
    if normalize:
        return drv.information_mutual_normalised(x, y, norm_factor='Y', cartesian_product=True)
    else:
        return drv.information_mutual(x, y, cartesian_product=True)


def mig(factors, codes, continuous_factors=True, nb_bins=10):
    ''' MIG metric from R. T. Q. Chen, X. Li, R. B. Grosse, and D. K. Duvenaud,
        “Isolating sources of disentanglement in variationalautoencoders,”
        in NeurIPS, 2018.

    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]

    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        # quantize values and get indexes
        factors = get_bin_index(factors, nb_bins)

    # quantize latent codes
    codes = minmax_scale(codes)  # normalize in [0, 1] all columns
    codes = get_bin_index(codes, nb_bins)  # quantize values and get indexes

    # compute mutual information matrix
    mi_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            mi_matrix[f, c] = get_mutual_information(
                factors[:, f], codes[:, c])

    # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        mi_f = np.sort(mi_matrix[f, :])
        # get diff between highest and second highest term and add it to total gap
        sum_gap += mi_f[-1] - mi_f[-2]

    # compute the mean gap
    mig_score = sum_gap / nb_factors

    return mig_score


# coding=utf-8
# Copyright 2018 Ubisoft La Forge Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def sap(factors, codes, continuous_factors=True, nb_bins=10, regression=True):
    ''' SAP metric from A. Kumar, P. Sattigeri, and A. Balakrishnan,
        “Variational inference of disentangled latent concepts from unlabeledobservations,”
        in ICLR, 2018.

    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    :param regression:                      True:   compute score using regression algorithms
                                            False:  compute score using classification algorithms
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]

    # perform regression
    if regression:
        assert (
            continuous_factors), f'Cannot perform SAP regression with discrete factors.'
        return _sap_regression(factors, codes, nb_factors, nb_codes)

    # perform classification
    else:
        # quantize factors if they are continuous
        if continuous_factors:
            factors = minmax_scale(factors)  # normalize in [0, 1] all columns
            # quantize values and get indexes
            factors = get_bin_index(factors, nb_bins)

        # normalize in [0, 1] all columns
        codes = minmax_scale(codes)

        # compute score using classification algorithms
        return _sap_classification(factors, codes, nb_factors, nb_codes)


def _sap_regression(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using regression algorithms

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    # compute R2 score matrix
    s_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            # train a linear regressor
            regr = LinearRegression()

            # train the model using the training sets
            regr.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))

            # make predictions using the testing set
            y_pred = regr.predict(codes[:, c].reshape(-1, 1))

            # compute R2 score
            r2 = r2_score(factors[:, f], y_pred)
            s_matrix[f, c] = max(0, r2)

    # compute the mean gap for all factors
    sum_gap = 0
    gap = []
    for f in range(nb_factors):
        # get diff between highest and second highest term and add it to total gap
        s_f = np.sort(s_matrix[f, :])
        sum_gap += s_f[-1] - s_f[-2]
        gap.append(s_f[-1] - s_f[-2])

    # compute the mean gap
    # sap_score = sum_gap / nb_factors

    gap = np.array(gap).reshape(-1, nb_factors//nb_codes)
    sap_score = np.mean(gap, axis=1)
    return sap_score


def _sap_classification(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using classification algorithms

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    # compute accuracy matrix
    s_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            # find the optimal number of splits
            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                # perform cross validation on the tree classifiers
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                scores = cross_val_score(
                    clf, codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1), cv=10)
                scores = scores.mean()

                if scores > best_score:
                    best_score = scores
                    best_sp = sp

            # train the model using the best performing parameter
            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            clf.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))

            # make predictions using the testing set
            y_pred = clf.predict(codes[:, c].reshape(-1, 1))

            # compute accuracy
            s_matrix[f, c] = accuracy_score(y_pred, factors[:, f])

    # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        # get diff between highest and second highest term and add it to total gap
        s_f = np.sort(s_matrix[f, :])
        sum_gap += s_f[-1] - s_f[-2]

    # compute the mean gap
    sap_score = sum_gap / nb_factors

    return sap_score


# Generate sample data
n_samples = 3000
n_components = 3

x, y = make_blobs(
    n_samples=n_samples,
    centers=n_components,
    cluster_std=0.60,
    random_state=0)


distance_matrix = pairwise_distances(x)

pres_at_5 = precision_at_k(distance_matrix, labels=y, k=5)
print(f"Precision at 5: {pres_at_5}")

mrr = mean_reciprocal_rank(distance_matrix, labels=y)
print(f"Mean Reciprocal Rank: {mrr}")

map = mean_average_precision(distance_matrix, labels=y)
print(f"Mean Average Precision: {map}")

dcg = ndcg(distance_matrix, labels=y)
print(f"NDCG: {dcg}")


# Generate sample data
n_attr = 4
n_samples = 3000
n_components = [2, 3, 4, 5]

x = []
y = []

for i in range(n_attr):
    x_, y_ = make_blobs(
        n_samples=n_samples, centers=n_components[i], cluster_std=0.60, random_state=0)
    x.append(x_)
    y.append(y_)

x = np.hstack(x)
y = np.vstack(y).T

# dic_ = dci(factors=x, codes=y, continuous_factors=True, model='random_forest')
# print(f"DCI: {dic_}")

# mig_ = mig(factors=x, codes=y, continuous_factors=True, nb_bins=10)
# print(f"MIG: {mig_}")

sap_ = sap(factors=x, codes=y, continuous_factors=True,
           nb_bins=10, regression=True)
print(f"SAP: {sap_}")
