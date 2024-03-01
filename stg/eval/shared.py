#!/usr/bin/env python3
"""Shared functionality between eval scripts."""
import random

import torch
import numpy as np


def determine_epochs(epochs: int or None, num_batches: int or None, batches_per_epoch: int) -> (int, int):
    """Determine number of epochs and batches per epoch."""
    if num_batches is not None:
        # Number of batches takes precedence over number of epochs
        epochs = num_batches // batches_per_epoch
    else:
        num_batches = epochs * batches_per_epoch
    return epochs, num_batches


def fix_seeds(seed: int):
    """Fix seed for reproducibility. TensorFlow seed must be set after importing TensorFlow."""
    # Fix PyTorch Seed for reproducibility
    torch.manual_seed(seed)
    # Fix random seed for reproducibility of splits
    random.seed(seed)
    # Fix numpy seed for reproducibility
    np.random.seed(seed)
