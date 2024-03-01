#!/usr/bin/env python3
"""Provides diverse helper functions."""
import logging
from collections import OrderedDict
from typing import Iterable

import math
import numpy as np
import pandas as pd
import torch.nn

log = logging.getLogger()


def compute_inverse_power(x: float):
    """Compute y such that 10^-y <= x <= 10^-(y-1)"""
    return math.ceil(-math.log10(x))


# Differential Privacy Helper Code
def compute_delta(n_train: int) -> float:
    """Compute delta such that delta = 10^-x <= 1/n_train.
    Setting delta to the inverse of the size of the training set is
    the best practice in literature.
    (e.g., https://www.cleverhans.io/privacy/2019/03/26/machine-learning-with-differential-privacy-in-tensorflow.html)
    Updated recommendation of 1/n^{1.1} according to
    [1] N. Ponomareva et al.,
    “How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy.”
    arXiv, Mar. 02, 2023. doi: 10.48550/arXiv.2303.00654.
    """
    # x = compute_inverse_power(1/n_train)
    # delta = 10 ** (-x)
    delta = 1 / (n_train ** 1.1)
    return delta


def count_parameters_torch(model: torch.nn.Module, print_layers: bool = True):
    """
    Compute the total number of parameters.
    :param model: PyTorch model
    :param print_layers: Print the parameters in each layer (Weights + Bias)
    :return: Total number of parameters
    """
    total = 0
    layers = OrderedDict()
    for name, p in model.named_parameters():
        parts = name.split('.')
        name = parts[0] + '.' + parts[1]
        if name in layers:
            layers[name] += p.numel()
        else:
            layers[name] = p.numel()
        total += p.numel()
    if print_layers:
        for layer in layers:
            log.info(f"{layer}: {layers[layer]}")
    return total


def find_bbox(
        df: pd.DataFrame,
        quantile: float = 1,
        x_label: str = 'lon',
        y_label: str = 'lat'
) -> (float, float, float, float):
    """Find a bounding box enclosing the defined quantile of points.

    :return: (Minimum X, Maximum X, Minimum Y, Minimum Y)
    """
    upper_quantiles = df.quantile(q=quantile, numeric_only=True)
    lower_quantiles = df.quantile(q=(1 - quantile), numeric_only=True)
    return lower_quantiles[x_label], upper_quantiles[x_label], \
        lower_quantiles[y_label], upper_quantiles[y_label]


def get_ref_point(series: np.ndarray or pd.Series) -> float or np.ndarray:
    """
    Get the reference point for normalization.
    :param series: Series or array of points
    :return: Reference point
    """
    # Convert to numpy array
    if type(series) is pd.Series or type(series) is pd.DataFrame:
        series = series.to_numpy()
    if len(series.shape) == 2:
        # nD points
        ref_point = (np.max(series, axis=0) + np.min(series, axis=0)) / 2
    else:
        # 1D points
        ref_point = (max(series) + min(series)) / 2
    return ref_point


def get_scaling_factor(series: np.ndarray or pd.Series, ref: float or Iterable[float]) -> float or np.ndarray:
    """
    Get the scale factor for normalization.
    :param series: Series or array of points
    :param ref: Reference point
    :return: Scale factor
    """
    if type(series) is pd.Series or type(series) is pd.DataFrame:
        series = series.to_numpy()
    if len(series.shape) == 2:
        # nD points
        scale_factor = np.max(abs(series - ref), axis=0)
    else:
        # 1D points
        scale_factor = max(abs(series - ref))
    return scale_factor


def dict2mdtable(d: dict, key: str = 'Name', val: str = 'Value') -> str:
    """
    Turn a dictionary into a Markdown table. Can be used to write hyperparameters into TensorBoard.
    Source: https://github.com/tensorflow/tensorboard/issues/46#issuecomment-1331147757
    :param d: Dictionary
    :param key: Title of Key column
    :param val: Title of Value colum
    :return: Markdown table as str
    """
    rows = [f'| {key} | {val} |']
    rows += ['|--|--|']
    rows += [f'| {k} | {v} |' for k, v in d.items()]
    return "  \n".join(rows)


def df2trajectory_dict(df: pd.DataFrame, tid_label: str = 'tid'):
    result = {tid: df for tid, df in df.groupby(tid_label)}
    return result
