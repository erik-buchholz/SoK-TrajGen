#!/usr/bin/env python3
"""Utility functions relating to data processing."""
import warnings
from typing import List, Union, Dict

import numpy as np
import pandas as pd


def normalize_sequence(series: np.ndarray or pd.Series,
                       ref: float = None,
                       scale_factor: float = None,
                       inplace: bool = False):
    """
    Normalizes a sequence of values to [-1;1]
    :param series: Series to normalize
    :param ref: (optional) Reference point (default: mean)
    :param scale_factor: (optional) Scale factor (default: max(abs(series)))
    :param inplace: Transform inplace
    :return:
    """
    # Check series type
    if type(series) is not np.ndarray and type(series) is not pd.Series:
        raise ValueError(f"Unexpected type '{type(series)}'.")
    if not inplace:
        series = series.astype(float, copy=True)
    else:
        series = series.astype(float, copy=False)
    if ref is None:
        ref = (max(series) + min(series)) / 2
    series -= ref
    if scale_factor is None:
        scale_factor = max(abs(series))
    series /= scale_factor
    return series, ref, scale_factor


def denormalize_sequence(series: np.ndarray or pd.Series,
                         ref: float,
                         scale_factor: float,
                         inplace: bool = False):
    """
    Denormalizes a sequence of values (inverse operation to normalize_sequence)
    :param series: Series to denormalize
    :param ref: float
    :param scale_factor: float
    :param inplace: Transform inplace
    :return:
    """
    # Check series type
    if type(series) is not np.ndarray and type(series) is not pd.Series:
        raise ValueError(f"Unexpected type '{type(series)}'.")
    if not inplace:
        series = series.copy()
    if scale_factor is not None:
        series *= scale_factor
    series += ref
    return series


def normalize_points(df: pd.DataFrame or np.ndarray,
                     columns: List[str or int] = ['x', 'y'],
                     ref: dict or list = None,
                     scale_factor: dict or list = None,
                     inplace: bool = False) -> (pd.DataFrame or np.ndarray,
                                                (float, float),
                                                (float, float)):
    """
    Scales dataframe columns to [-1;1]
    :param df: MUST contain columns x_col and y_col
    :param columns: List of columns or array indices to normalize
    :param ref: (optional) Reference points as list in column order or dict with columns as keys (default: mean)
    :param scale_factor: (optional) Scale factors as list in column order or  dict with columns as keys (default: max(abs(series)))
    :param inplace: Transform inplace
    :return: df, List[ref_points], List[scale_factor] (list are in same order als 'columns')
    """
    # Check series type
    if not isinstance(df, pd.DataFrame) and not isinstance(df, np.ndarray):
        raise ValueError(f"Unexpected type '{type(df)}'.")

    # Make a copy
    if not inplace:
        df = df.copy()

    # Convert to float
    if isinstance(df, pd.DataFrame):
        df[columns] = df[columns].astype(float)
    else:
        df = df.astype(float, copy=False)  # copied before

    references = []
    scale_factors = []

    # Dummy values for ref and scale_factor
    if ref is None:
        ref = {col: None for col in columns}
    if scale_factor is None:
        scale_factor = {col: None for col in columns}

    for i, col in enumerate(columns):
        r = ref[col] if isinstance(ref, dict) else ref[i]
        s = scale_factor[col] if isinstance(scale_factor, dict) else scale_factor[i]
        if isinstance(df, pd.DataFrame):
            df[col], r, s = normalize_sequence(df[col], ref=r, scale_factor=s,
                                               inplace=True)  # inplace b/c operating on copy
            assert (abs(df[col]) <= 1.0 + 1e-5).all(), f"Problem for column '{col}': {df[col]}."
        elif isinstance(df, np.ndarray):
            df[:, col], r, s = normalize_sequence(df[:, col], ref=r, scale_factor=s,
                                                  inplace=True)  # inplace b/c operating on copy
            assert (abs(df[:, col]) <= 1.0 + 1e-5).all(), f"Problem for column '{col}': {df[col]}."
        else:
            raise ValueError(f"Unexpected type '{type(df)}'.")
        references.append(r)
        scale_factors.append(s)

    return df, references, scale_factors


def denormalize_points(df: Union[np.ndarray, pd.Series, pd.DataFrame],
                       ref: Union[List[float], Dict],
                       scale: Union[List[float], Dict],
                       columns: Union[List[Union[str, int]], List[int], List[str]] = (0, 1),
                       inplace: bool = False) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Denormalizes a dataframe (inverse operation to normalize_points)
    :param df: List of points to denormalize
    :param ref: reference point either as list or dict with columns as keys
    :param scale: scale factor either as list or dict with columns as keys
    :param columns: List of columns or array indices to denormalize
    :param inplace: Transform inplace
    :return: denormalized df
    """
    # Check series type
    if not isinstance(df, (np.ndarray, pd.Series, pd.DataFrame)):
        raise ValueError(f"Unexpected type '{type(df)}'.")
    if not inplace:
        df = df.astype(float, copy=True)
    else:
        df = df.astype(float, copy=False)

    for i, col in enumerate(columns):
        if isinstance(df, pd.DataFrame):
            # Use .loc for label-based indexing if 'df' is a DataFrame
            df.loc[:, col] = denormalize_sequence(df.loc[:, col], ref[i], scale[i], inplace=True)
        else:
            # Assuming 'df' is a NumPy array if not a DataFrame
            df[:, col] = denormalize_sequence(df[:, col], ref[i], scale[i], inplace=True)

    return df
