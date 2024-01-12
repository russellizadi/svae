#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Russell Izadi 2023
"""

import glob
import os
from pathlib import Path
from typing import Tuple, Union

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import random


class VOCALSET(Dataset):
    def __init__(self, root: Union[str, Path],
                 split: str = None,
                 seed: int = None,
                 ratio: float = None) -> None:
        list_paths = glob.glob(os.path.join(
            root, '**/*.wav'), recursive=True)

        if split is not None:
            list_paths_train, list_paths_valid = self.random_split(
                list_paths, seed=seed, ratio=ratio)
            if split == 'train':
                list_paths = list_paths_train
            elif split == 'valid':
                list_paths = list_paths_valid

        self.list_paths = list_paths
        self.data = self.get_data(list_paths)
        self.indices = self.get_indices(self.data)

    def __getitem__(self, i: int) -> Tuple[Tensor, int, set, str]:
        if i >= len(self):
            raise IndexError
        path_wav, label = self.data[i]
        return path_wav, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data(list_path_wavs):

        data = []
        for path in list_path_wavs:
            singer = path.split("/")[-4]
            context = path.split("/")[-3]
            technique = path.split("/")[-2]
            vowel = path.split("/")[-1].split(".")[0].split("_")[-1]

            label = {
                'singer': singer,
                'context': context,
                'technique': technique,
                'vowel': vowel}

            data.append((path, label))

        return data

    @staticmethod
    def get_indices(data):

        indices = {}
        for idx, (_, label) in enumerate(data):
            for key, value in label.items():
                if key not in indices:
                    indices[key] = {}
                if value not in indices[key]:
                    indices[key][value] = []
                indices[key][value].append(idx)

        return indices

    @staticmethod
    def random_split(list_, seed=0, ratio=0.8):
        random.seed(seed)
        random.shuffle(list_)
        n_samples = int(len(list_) * ratio)
        return list_[:n_samples], list_[n_samples:]


if __name__ == '__main__':
    dataset = VOCALSET(
        root="/ext/projects/disentangle/datasets/vocalset")
    print("Number of samples: ", len(dataset))
    for sample in dataset:
        path_wav, label = sample
        waveform, sample_rate = torchaudio.load(path_wav)
        print(waveform.shape, sample_rate, label)
        break
