import math

import numpy as np
import pandas as pd
import logging
import time
import torch
import argparse

from datetime import datetime

from deep_squeeze.autoencoder import AutoEncoder
from deep_squeeze.preprocessing import ds_preprocessing
from deep_squeeze.train_loop import train
from deep_squeeze.materialization import materialize, materialize_with_post_binning, \
    materialize_with_bin_difference
from deep_squeeze.disk_storing import store_on_disk, calculate_compression_ratio
from deep_squeeze.experiment import repeat_n_times, display_compression_results, run_full_experiments, \
    run_scaling_experiment, baseline_compression_ratios
from deep_squeeze.bayesian_optimizer import minimize_comp_ratio


def checker():
    original = pd.read_csv("data/corel_processed.csv", header=None)
    reconstruction = pd.read_csv("data/output.csv", header=None)
    range = original.max() - original.min()
    percentages = (original-reconstruction).abs().divide(range) * 100
    print("Max error threshold percentages per column: ")
    print(percentages.max().to_numpy())


if __name__ == '__main__':
    checker()
