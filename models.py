#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Russell Izadi 2023
"""

import math
from os import path
from turtle import st
# from unittest import result
# from py import log

import torch
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        d_input,
        d_hidden,
        d_latent,
        rates,
        n_heads,
        n_layers,
        max_duration,
    ):
        super().__init__()

        self.rates = rates
        self.d_hidden = d_hidden

        self.in_pos_emb = nn.Embedding(max_duration, d_hidden)
        self.latent_pos_emb = nn.Embedding(max_duration, d_hidden)
        self.latent_attr_emb = nn.Embedding(len(rates), d_hidden)

        self.in_layer = nn.Linear(d_input, d_hidden)
        self.latent_layer = nn.Linear(d_hidden, d_latent)

        enc_layer = TransformerDecoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=True,
            device=None,
            dtype=None)
        self.enc_layers = TransformerDecoder(
            decoder_layer=enc_layer,
            num_layers=n_layers,
            norm=None)

    def forward(self, x, mask=None):
        # x: (?, T, d_input), mask: (?, T)
        in_duration = x.shape[1]

        # Input layer: (?, T, d_input) -> (?, T, d_hidden)
        x = self.in_layer(x)

        # Add positional embeddings
        in_pos = torch.LongTensor(list(range(in_duration))).to(x.device)
        # .expand(-1, self.d_hidden)  # (?, T, d_hidden)
        x = x + self.in_pos_emb(in_pos)

        # Init output with positional and attribute embeddings
        durations = []
        out_attr = []
        out_pos = []
        for i_attr, rate in enumerate(self.rates):
            duration_i = max(int(in_duration * rate), 1)
            durations.append(duration_i)
            out_attr.extend(duration_i * [i_attr])
            out_pos.extend(list(range(duration_i)))

        out_attr = torch.LongTensor(out_attr).to(x.device)
        out_pos = torch.LongTensor(out_pos).to(x.device)
        z = self.latent_attr_emb(out_attr) + self.latent_pos_emb(out_pos)
        z = z.expand(x.shape[0], -1, -1)  # (?, T, d_hidden)

        # Encode: -> (?, L, d_hidden)
        # L = sum(durations)
        z = self.enc_layers(
            tgt=z,
            memory=x,
            memory_key_padding_mask=mask)

        # Output layer: (?, L, d_latent) -> (?, L, d_latent)
        z = self.latent_layer(z)

        # z: (?, L, d_latent), durations: (n_attrs, )
        return z, durations


class Decoder(nn.Module):
    def __init__(
        self,
        d_latent,
        d_hidden,
        d_output,
        n_attrs,
        n_heads,
        n_layers,
        max_duration,
    ):
        super().__init__()

        self.d_hidden = d_hidden

        self.latent_attr_emb = nn.Embedding(n_attrs, d_hidden)
        self.latent_pos_emb = nn.Embedding(max_duration, d_hidden)
        self.out_pos_emb = nn.Embedding(max_duration, d_hidden)

        self.latent_layer = nn.Linear(d_latent, d_hidden)
        self.out_layer = nn.Linear(d_hidden, d_output)

        dec_layer = TransformerDecoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=True,
            device=None,
            dtype=None)
        self.dec_layers = TransformerDecoder(
            decoder_layer=dec_layer,
            num_layers=n_layers,
            norm=None)

    def forward(self, z, mask, durations):
        # z: (?, L, d_latent), mask: (?, T), durations: (n_attrs, )
        out_duration = mask.shape[-1]

        # Latent layer: (?, L, d_latent) -> (?, L, d_hidden)
        z = self.latent_layer(z)

        # Add positional and attribute embeddings
        latent_attr = []
        latent_pos = []
        for i_attr, duration_i in enumerate(durations):
            latent_attr.extend(duration_i * [i_attr])
            latent_pos.extend(list(range(duration_i)))

        latent_attr = torch.LongTensor(latent_attr).to(z.device)
        latent_pos = torch.LongTensor(latent_pos).to(z.device)

        z = z + self.latent_attr_emb(latent_attr) + \
            self.latent_pos_emb(latent_pos)

        # Init output with positional embeddings
        out_pos = torch.LongTensor(list(range(out_duration))).to(z.device)
        y = self.out_pos_emb(out_pos)
        y = y.expand(z.shape[0], -1, -1)  # (?, T, d_hidden)

        # Mask future positions: (T, T)
        out_mask = torch.triu(torch.full((out_duration, out_duration), float(
            '-inf'), device=z.device), diagonal=1).bool()

        # Decode: -> (?, T, d_hidden)
        y = self.dec_layers(
            tgt=y,
            memory=z,
            tgt_mask=out_mask,
            tgt_key_padding_mask=mask)

        # Output layer: (?, T, d_hidden) -> (?, T, d_output)
        y = self.out_layer(y)
        return y


class GaussianPrior(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.register_buffer("loc", torch.zeros(dim))
        self.register_buffer("scale", math.log(math.e - 1) * torch.ones(dim))

    def log_prob(self, z):
        # x: (?, L, d_latent)
        scale = F.softplus(self.scale - 1e-4) + 1e-4
        return torch.distributions.Normal(self.loc, scale).log_prob(z).sum(-1)

    def forward(self, x):
        return x


class GaussianPosteriorQ(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        d_cond = 2 * dim

        d_hidden = math.ceil(math.sqrt(2*dim*(d_cond))*2)
        self.param_layer = nn.Sequential(
            nn.Linear(d_cond, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, dim*2))

    def rsample(self, c, out_prob=False):
        loc, scale = self.param_layer(c).chunk(2, dim=-1)
        scale = F.softplus(scale - 1e-4) + 1e-4

        dist = torch.distributions.Normal(loc, scale)
        result = dist.rsample()

        if not out_prob:
            return result
        else:
            return result, dist.log_prob(result).sum(-1)

    def forward(self, x):
        return x


class GaussianPosteriorP(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def rsample(self, x, param, out_prob=False):

        dist = torch.distributions.Normal(
            loc=param, scale=torch.ones_like(param))
        result = dist.mean

        if not out_prob:
            return result
        else:
            return result, dist.log_prob(x).sum(-1)

    def forward(self, x):
        return x


class VAE(nn.Module):
    def __init__(
            self,
            encoder, decoder,
            prior,
            posterior_q,
            posterior_p):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.posterior_q = posterior_q
        self.posterior_p = posterior_p

    @staticmethod
    def aggregate(c, attr):
        return c

    def forward(self, x, mask, attr=None):
        """
        Args:
            x (torch.Tensor): (B, L, D) data
            mask (torch.Tensor): (B, L) data: False, padding: True
            attr (torch.Tensor): (B, n_attrs)

        Returns:
            results (dict): {
                'loss': (1),
                'loss_recon': (1),
                'loss_kl': (1),
                'y': (B, L, D),
                'z': (B, n_attrs, z_dim)
            }
        """
        assert x.ndim == 3 and mask.ndim == 2, (x.shape, mask.shape)

        # encode
        c, durations = self.encoder(x, mask)

        if self.training:
            c = self.aggregate(c, attr)

        # (B, n_attrs, z_dim), (B, n_attrs)
        z, log_q_z_given_x = self.posterior_q.rsample(c, out_prob=True)
        log_p_z = self.prior.log_prob(z)  # (B, n_attrs)

        param = self.decoder(z, mask, durations)
        # (B, L, d_input), (B, L)
        y, log_p_x_given_z = self.posterior_p.rsample(x, param, out_prob=True)

        kl = log_q_z_given_x - log_p_z  # (B, L)
        kl = kl.sum(1)  # (B)

        unmasked = 1. - mask.float()
        n_unmasked = unmasked.sum(1)  # (B)

        log_p_x_given_z = log_p_x_given_z * unmasked
        log_p_x_given_z = log_p_x_given_z.sum(1)  # (B)

        logp = log_p_x_given_z - kl  # (B)
        logp = logp / n_unmasked  # (B)
        loss = - logp.mean()  # (1)

        loss_recon = - (log_p_x_given_z / n_unmasked).mean()  # (1)
        loss_kl = (kl / n_unmasked).mean()  # (1)

        y = y * unmasked.unsqueeze(2)

        results = {
            'loss': loss,
            'loss_recon': loss_recon,
            'loss_kl': loss_kl,
            'y': y,
            'z': z}
        return results


class MLVAE(VAE):
    def __init__(
            self,
            encoder, decoder,
            prior,
            posterior_q,
            posterior_p):
        super().__init__(
            encoder,
            decoder,
            prior,
            posterior_q,
            posterior_p)

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.posterior_q = posterior_q
        self.posterior_p = posterior_p

    @staticmethod
    def aggregate(c, attr):
        assert attr is not None
        b = c.shape[0] // 2
        loc, scale = c.chunk(2, dim=-1)
        scale_inv = 1. / (scale + 1.e-5)
        sum_scale_inv = scale_inv[:b][attr[:b]] + scale_inv[b:][attr[b:]]
        sum_loc_scale = (loc[:b][attr[:b]] * scale_inv[:b][attr[:b]] +
                         loc[b:][attr[b:]] * scale_inv[b:][attr[b:]])
        new_loc = sum_loc_scale / sum_scale_inv
        new_scale = 1. / sum_scale_inv
        new_c = torch.cat([new_loc, new_scale], dim=-1)
        c[:b][attr[:b]] = new_c
        c[b:][attr[b:]] = new_c
        return c


class GVAE(VAE):
    def __init__(
            self,
            encoder, decoder,
            prior,
            posterior_q,
            posterior_p):
        super().__init__(
            encoder,
            decoder,
            prior,
            posterior_q,
            posterior_p)

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.posterior_q = posterior_q
        self.posterior_p = posterior_p

    @staticmethod
    def aggregate(c, attr):
        assert attr is not None
        b = c.shape[0] // 2
        avg = (c[:b][attr[:b]] + c[b:][attr[b:]]) / 2.
        c[:b][attr[:b]] = avg
        c[b:][attr[b:]] = avg
        return c


class SVAE(VAE):
    def __init__(
            self,
            encoder,
            decoder,
            prior,
            posterior_q,
            posterior_p):
        super().__init__(
            encoder,
            decoder,
            prior,
            posterior_q,
            posterior_p)

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.posterior_q = posterior_q
        self.posterior_p = posterior_p

    @staticmethod
    def aggregate(c, attr):
        assert attr is not None
        b = c.shape[0] // 2
        c[:b][attr[:b]] = c[b:][attr[b:]]
        c[b:][attr[b:]] = c[:b][attr[:b]]
        return c
