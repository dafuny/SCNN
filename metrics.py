#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:22:26 2019

@author: luoyao
"""

from sklearn.model_selection import KFold
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def select_threshold(distances, matches, thresholds):
    best_threshold_true_predicts = 0
    best_threshold = 0
    for threshold in thresholds:
        true_predicts = torch.sum((
            distances < threshold
        ) == matches)

        if true_predicts > best_threshold_true_predicts:
            best_threshold_true_predicts = true_predicts
            best_threshold = threshold

    return best_threshold


def compute_roc(distances, matches, thresholds, fold_size=10):
    assert(len(distances) == len(matches))

    kf = KFold(n_splits=fold_size, shuffle=False)

    tpr = torch.zeros(fold_size, len(thresholds))
    fpr = torch.zeros(fold_size, len(thresholds))
    accuracy = torch.zeros(fold_size)
    best_thresholds = []

    for fold_index, (training_indices, val_indices) \
            in enumerate(kf.split(range(len(distances)))):

        training_distances = distances[training_indices]
        training_matches = matches[training_indices]

        # 1. find the best threshold for this fold using training set
        best_threshold_true_predicts = 0
        for threshold_index, threshold in enumerate(thresholds):
            true_predicts = torch.sum((
                training_distances < threshold
            ) == training_matches)

            if true_predicts > best_threshold_true_predicts:
                best_threshold = threshold
                best_threshold_true_predicts = true_predicts

        # 2. calculate tpr, fpr on validation set
        val_distances = distances[val_indices]
        val_matches = matches[val_indices]
        for threshold_index, threshold in enumerate(thresholds):
            predicts = val_distances < threshold

            tp = torch.sum(predicts & val_matches).item()
            fp = torch.sum(predicts & ~val_matches).item()
            tn = torch.sum(~predicts & ~val_matches).item()
            fn = torch.sum(~predicts & val_matches).item()

            tpr[fold_index][threshold_index] = float(tp) / (tp + fn)
            fpr[fold_index][threshold_index] = float(fp) / (fp + tn)

        best_thresholds.append(best_threshold)
        accuracy[fold_index] = best_threshold_true_predicts.item() / float(
            len(training_indices))

    # average fold
    tpr = torch.mean(tpr, dim=0).numpy()
    fpr = torch.mean(fpr, dim=0).numpy()
    accuracy = torch.mean(accuracy, dim=0).item()

    return tpr, fpr, accuracy, best_thresholds

def generate_roc_curve(fpr, tpr, path):
    assert len(fpr) == len(tpr)

    fig = plt.figure()
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.title('SET1', fontname = "Times New Roman")         #add by ly
    plt.xlabel('False Positive Rate', fontname = "Times New Roman")
    plt.ylabel('True Positive Rate', fontname = "Times New Roman")
    plt.grid(linestyle = '--')
    plt.plot(fpr, tpr, color='deepskyblue')
    fig.savefig(path, dpi=300)

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds = 10, pca = 0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        # dist = pdist(np.vstack([embeddings1, embeddings2]), 'cosine')

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print("doing pca on", fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis = 0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components = pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
#         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds