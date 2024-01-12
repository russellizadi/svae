#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Russell Izadi 2023
"""

import torchaudio
from collections import defaultdict
import csv
import numpy as np

from vocalset import VOCALSET

if __name__ == '__main__':
    dataset = VOCALSET(
        root="/ext/projects/disentangle/datasets/vocalset")

    # Initialize variables to store the required values
    # waveform_shapes = []
    # sample_rates = set()
    # labels = defaultdict(set)
    csvfile = open("vocalset.csv", "w", newline="")
    writer = csv.writer(csvfile)

    for sample in dataset:
        path_wav, label = sample
        row = []
        row.append(path_wav)

        waveform, sample_rate = torchaudio.load(path_wav)
        row.append(sample_rate)
        row.append(waveform.shape[0])
        row.append(waveform.shape[-1])
        row.append(waveform.shape[-1]/sample_rate)

        # Get the tensor shape
        # waveform_shapes.append(waveform.shape[-1])

        # # Add the integer value to the set
        # sample_rates.add(sample_rate)

        # Add dictionary values to the corresponding sets for each key

        for key, value in label.items():
            row.append(value)
            # if key == 'technique' and value == 'vibrado':
            #     print(path_wav)
            # labels[key].add(value)

        # Write the row to the CSV file
        writer.writerow(row)

    # Close the CSV file
    csvfile.close()

    # assert len(sample_rates) == 1, "Sample rates should be the same"
    # sample_rate = sample_rates.pop()

    # # Calculate basic statistics
    # info = {
    #     'sample_rate': sample_rate,
    #     'mean-shape': np.mean(waveform_shapes, axis=0),
    #     'std-shape': np.std(waveform_shapes, axis=0),
    #     'min-shape': np.min(waveform_shapes, axis=0),
    #     'max-shape': np.max(waveform_shapes, axis=0),
    #     'mean-duration': np.mean(waveform_shapes) / sample_rate,
    #     'std-duration': np.std(waveform_shapes) / sample_rate,
    #     'min-duration': np.min(waveform_shapes) / sample_rate,
    #     'max-duration': np.max(waveform_shapes) / sample_rate,
    # }

    # print(info)
    # print(labels)
