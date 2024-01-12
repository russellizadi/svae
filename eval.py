#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Russell Izadi 2023
"""

from json import decoder
from vocalset import VOCALSET
import argparse
import utils
import os
import torchaudio
import torch
import models
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
# from torch.optim.lr_scheduler import MultiStepLR
from sklearn.manifold import TSNE

from train import get_model, get_loaders, get_datasets


def main(args):

    # Initialize the experiment
    if args.wandb is True:
        wandb.login()
        run = wandb.init(config=args,
                         project=args.project,
                         entity=args.entity,
                         dir=args.dir,
                         notes=args.notes,
                         tags=args.tags)

    # Get the dataset
    _, valid_loader = get_loaders(args.data)

    # Get the model
    model = get_model(args.model)
    model = model.to(args.device)

    # Load the checkpoint
    checkpoint = torch.load(args.model.path)

    # Load the model state dict from the checkpoint
    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model
    latents_list = []
    labels_list = []
    for i_batch, batch in enumerate(valid_loader):
        features, masks, labels = batch

        features, masks = features.to(
            args.device), masks.to(args.device)
        results = model(features, masks)

        latent = results['z']
        latents_list.append(
            latent.detach().cpu().numpy().reshape(latent.shape[0], -1))

        labels_list.extend(
            [[label[attr] for attr in args.data.attrs] for label in labels])

    labels_list = [list(items) for items in zip(*labels_list)]
    labels_list = [
        utils.str_to_int(labels) for labels in labels_list]

    latents = np.vstack(latents_list)  # (?, d_latent*n_attrs)
    labels = np.array(labels_list).T  # (?, n_attrs)

    valid_log_data = {}
    d_latent = args.model.decoder.d_latent
    for i_attr, attr in enumerate(args.data.attrs):
        latents_i = latents[:, d_latent*i_attr:d_latent*(i_attr+1)]

        labels_i = labels[:, i_attr]

        # Compute the metrics
        distance_matrix = pairwise_distances(latents_i)
        valid_log_data[f"valid/prec@5/{attr}"] = utils.precision_at_k(
            distance_matrix, labels=labels_i, k=5)
        valid_log_data[f"valid/mrr/{attr}"] = utils.mean_reciprocal_rank(
            distance_matrix, labels=labels_i)
        valid_log_data[f"valid/map/{attr}"] = utils.mean_average_precision(
            distance_matrix, labels=labels_i)
        valid_log_data[f"valid/ndcg/{attr}"] = utils.ndcg(
            distance_matrix, labels=labels_i)

        print(attr)
        print(" & ".join((
            f"{args.model.name}".upper(),
            f"{valid_log_data[f'valid/prec@5/{attr}']:.2f}",
            f"{valid_log_data[f'valid/mrr/{attr}']:.2f}",
            f"{valid_log_data[f'valid/ndcg/{attr}']:.2f}",
            f"{valid_log_data[f'valid/map/{attr}']:.2f}")))
        print()

        # Plot the scatter plot
        if d_latent > 2:
            # Reduce the dimensionality of the latent space
            dr = TSNE(n_components=2)
            latents_i = dr.fit_transform(latents_i)

        fig = plt.figure(figsize=(5, 5))
        unique_labels = np.unique(labels_i)
        for label in unique_labels:
            indices = (labels_i == label)
            plt.scatter(latents_i[indices, 0],
                        latents_i[indices, 1], label=str(label))

        valid_log_data[f"valid/latents/{attr}"] = wandb.Image(fig)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(h_pad=0, w_pad=0)
        for suffix in ['png', 'svg', 'pdf']:
            fig_path = os.path.join(run.dir, f'{attr}.{suffix}')
            plt.savefig(fig_path, dpi=100)
        plt.close(fig)


if __name__ == "__main__":

    # Get the command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yml", type=str,
                        metavar='c', help='Path to the config file.')
    args = parser.parse_args()

    # Load the config file
    args = utils.read_yml(args.config)

    main(args)
