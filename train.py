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


def get_datasets(args):

    def resample(x, orig_freq):
        x = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=args.sample_rate)(x)
        return x

    def mel_spec(x):
        x = torchaudio.transforms.MelSpectrogram(**args.features)(x)
        x = torchaudio.transforms.AmplitudeToDB()(x)
        return x

    def feature_fn(path_wav):
        waveform, sample_rate = torchaudio.load(path_wav)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = resample(waveform, sample_rate)
        feature = mel_spec(waveform)
        return feature

    def train_prep_fn(sample):
        path_wav, path_wav_, attr = sample
        feature = feature_fn(path_wav)
        feature_ = feature_fn(path_wav_)
        return feature, feature_, attr

    def valid_prep_fn(sample):
        path_wav, label = sample
        feature = feature_fn(path_wav)
        return feature, label

    def train_norm_fn(sample):
        feature, feature_, attr = sample
        feature = feature.sub_(args.mean).div_(args.std)
        feature_ = feature_.sub_(args.mean).div_(args.std)
        return feature, feature_, attr

    def valid_norm_fn(sample):
        feature, label = sample
        feature = feature.sub_(args.mean).div_(args.std)
        return feature, label

    # Get the train
    train_ds = VOCALSET(split="train", **args.vocalset)
    train_ds_list = []
    for attr in args.attrs:
        train_ds_attr = utils.Pair(train_ds, attr)
        train_ds_attr = utils.Transform(train_ds_attr, train_prep_fn)
        train_ds_list.append(train_ds_attr)
    train_ds = torch.utils.data.ConcatDataset(train_ds_list)

    # Get the valid
    valid_ds = VOCALSET(split="valid", **args.vocalset)
    valid_ds_list = []
    for attr in args.attrs:
        valid_ds_attr = utils.Transform(valid_ds, valid_prep_fn)
        valid_ds_list.append(valid_ds_attr)
    valid_ds = torch.utils.data.ConcatDataset(valid_ds_list)

    # Normalize the dataset
    if args.norm is True:
        mean, std = utils.dataset_mean_std(train_ds)
        args.mean = mean
        args.std = std

    train_ds = utils.Transform(train_ds, train_norm_fn)
    valid_ds = utils.Transform(valid_ds, valid_norm_fn)

    return train_ds, valid_ds


def get_loaders(args):

    def train_collate_fn(batch):
        features, features_, attrs = zip(*batch)
        features += features_
        attrs += attrs

        attrs = torch.tensor([[int(item == ref_item)
                             for ref_item in args.attrs] for item in attrs], dtype=torch.bool)
        features = [feature.squeeze(0).transpose(0, 1) for feature in features]
        lengths = [len(feature) for feature in features]
        features = pad_sequence(features, batch_first=True)
        duration_range = torch.arange(features.size(1))
        masks = torch.stack([duration_range > length for length in lengths])
        return features, masks, attrs

    def valid_collate_fn(batch):
        features, labels = zip(*batch)

        features = [feature.squeeze(0).transpose(0, 1) for feature in features]
        lengths = [len(feature) for feature in features]
        features = pad_sequence(features, batch_first=True)
        duration_range = torch.arange(features.size(1))
        masks = torch.stack([duration_range > length for length in lengths])
        return features, masks, labels

    train_ds, valid_ds = get_datasets(args)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        collate_fn=train_collate_fn,
        **args.loader)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_ds,
        collate_fn=valid_collate_fn,
        **args.loader)

    return train_loader, valid_loader


def get_model(args):
    encoder = models.Encoder(**args.encoder)
    decoder = models.Decoder(**args.decoder)
    prior = models.GaussianPrior(**args.prior)
    posterior_q = models.GaussianPosteriorQ(**args.posterior_q)
    posterior_p = models.GaussianPosteriorP(**args.posterior_p)

    if args.name == "vae":
        model = models.VAE(encoder, decoder, prior, posterior_q, posterior_p)
    elif args.name == "mlvae":
        model = models.MLVAE(encoder, decoder, prior, posterior_q, posterior_p)
    elif args.name == "gvae":
        model = models.GVAE(encoder, decoder, prior, posterior_q, posterior_p)
    elif args.name == "svae":
        model = models.SVAE(encoder, decoder, prior, posterior_q, posterior_p)

    return model


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
    train_loader, valid_loader = get_loaders(args.data)
    if args.wandb is True:
        wandb.config.update(args, allow_val_change=True)
        wandb.config.update(
            {
                "train_loader_size": len(train_loader.dataset),
                "valid_loader_size": len(valid_loader.dataset),
            }
        )

    model = get_model(args.model)
    model = model.to(args.device)

    if args.wandb is True:
        wandb.watch(
            model,
            criterion=None,
            log='all',
            log_freq=args.watch_freq)

        # Get the number of parameters
        total_params, trainable_params = utils.num_model_params(model)
        wandb.config.update(
            {
                "total_params": total_params,
                "trainable_params": trainable_params
            }
        )

    optimizer = torch.optim.Adam(
        model.parameters(), **args.engine.optim)

    # Log params and gradients
    if args.wandb is True and args.watch_freq > 0:
        wandb.watch(
            model,
            criterion=None,
            log='all',
            log_freq=args.watch_freq,
            log_graph=False)

    # Train/Valid
    step = 0
    epoch = 0
    while step <= args.engine.steps:
        epoch += 1
        if 0 < args.engine.debug < step:
            break

        # Train
        model.train()
        for batch in train_loader:
            step += 1

            optimizer.zero_grad()
            features, masks, attrs = batch
            features, masks = features.to(args.device), masks.to(args.device)
            results = model(features, masks, attrs)

            # print(features.min().item(),
            #       features.max().item(),
            #       features.mean().item(),
            #       features.std().item(),
            #       features.shape)

            loss = results["loss"]
            loss.backward()
            optimizer.step()

            # Log
            train_log_data = {
                "train/loss/loss": loss.item(),
                "train/loss/kl": results['loss_kl'].item(),
                "train/loss/recon": results['loss_recon'].item(),
            }
            if args.wandb is True:
                wandb.log(data=train_log_data, step=step)

        # Validate
        if epoch == 1 or epoch % args.valid_freq == 0:
            model.eval()
            features_list = []
            latents_list = []
            labels_list = []
            recons_list = []
            losses_list = []

            for i_batch, batch in enumerate(valid_loader):
                features, masks, labels = batch

                if i_batch == 0:
                    features_list.append(
                        features.detach().cpu().numpy())

                features, masks = features.to(
                    args.device), masks.to(args.device)
                results = model(features, masks)

                latent = results['z']
                latents_list.append(
                    latent.detach().cpu().numpy().reshape(latent.shape[0], -1))

                labels_list.extend(
                    [[label[attr] for attr in args.data.attrs] for label in labels])

                if i_batch == 0:
                    recons = results['y']
                    recons_list.append(
                        recons.detach().cpu().numpy())

                losses_list.append(
                    results['loss'].detach().cpu().numpy())

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
                valid_log_data[f"valid/prec@1/{attr}"] = utils.precision_at_k(
                    distance_matrix, labels=labels_i, k=1)
                valid_log_data[f"valid/mrr/{attr}"] = utils.mean_reciprocal_rank(
                    distance_matrix, labels=labels_i)
                valid_log_data[f"valid/map/{attr}"] = utils.mean_average_precision(
                    distance_matrix, labels=labels_i)
                valid_log_data[f"valid/ndcg/{attr}"] = utils.ndcg(
                    distance_matrix, labels=labels_i)

                # Plot the scatter plot
                if d_latent > 2:
                    # Reduce the dimensionality of the latent space
                    # dr = PCA(n_components=2)
                    dr = TSNE(n_components=2)
                    latents_i = dr.fit_transform(latents_i)

                fig = plt.figure()
                unique_labels = np.unique(labels_i)
                for label in unique_labels:
                    indices = (labels_i == label)
                    plt.scatter(latents_i[indices, 0],
                                latents_i[indices, 1], label=str(label))
                plt.legend()
                valid_log_data[f"valid/latents/{attr}"] = wandb.Image(fig)
                plt.close(fig)

            # Plot the reconstructions
            fig = plt.figure()
            img_list = []
            for i_sample, (features, recons) in enumerate(
                    zip(features_list[0], recons_list[0])):
                if i_sample >= 8:
                    break
                # features, recons: (T x D)
                features = np.flip(features, axis=-1).T
                recons = np.flip(recons, axis=-1).T
                img = np.concatenate([features, recons], axis=0)
                label_i = [f"{attr}: {label}" for attr, label in
                           zip(args.data.attrs, labels[i_sample])]
                plt.imshow(img)
                plt.title(" | ".join(label_i))
                img_list.append(wandb.Image(fig))
            valid_log_data["valid/recons/"] = img_list
            plt.close(fig)

            # Save the model
            model_path = os.path.join(run.dir, 'model.pth')
            torch.save(model.state_dict(), model_path)

            if args.wandb is True:
                wandb.log(data=valid_log_data, step=step)


if __name__ == "__main__":

    # Get the command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="train.yml", type=str,
                        metavar='c', help='Path to the config file.')
    args = parser.parse_args()

    # Load the config file
    args = utils.read_yml(args.config)

    main(args)
