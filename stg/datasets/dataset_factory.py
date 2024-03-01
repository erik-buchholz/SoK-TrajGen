#!/usr/bin/env python3
""" """
from enum import Enum

from stg.datasets import DatasetModes, TrajectoryDataset, FSNYCDataset, GeoLifeDataset, mnist_sequential


class Datasets(str, Enum):
    FS = 'fs'
    GEOLIFE = 'geolife'
    MNIST_SEQUENTIAL = 'mnist_sequential'
    MNIST = 'mnist_sequential'  # Alias to MNIST_SEQUENTIAL


def get_dataset(
        dataset_name: Datasets,
        mode: DatasetModes = DatasetModes.ALL,
        latlon_only: bool = False,
        normalize: bool = True,
        return_labels: bool = False,
        keep_original: bool = True,
        sort: bool = False,
) -> TrajectoryDataset:
    if dataset_name == Datasets.FS:
        # FS-NYC
        dataset: TrajectoryDataset = FSNYCDataset(
            mode=mode,
            latlon_only=latlon_only,
            normalize=normalize,
            return_labels=return_labels,
            keep_original=keep_original,
            sort=sort
        )
    elif dataset_name == Datasets.GEOLIFE:
        dataset: TrajectoryDataset = GeoLifeDataset(
            mode=mode,
            latlon_only=latlon_only,
            normalize=normalize,
            return_labels=return_labels,
            keep_original=keep_original,
            sort=sort
        )
    elif dataset_name == Datasets.MNIST_SEQUENTIAL:
        if not normalize or not return_labels:
            raise ValueError("MNIST dataset only supports normalize=True and return_labels=True.")
        return mnist_sequential(dim=28)
    else:
        raise NotImplementedError(f"Unsupported Dataset: {str(dataset_name)}")
    return dataset
