#!/usr/bin/env python3
"""This module contains wrappers for the datasets and data loaders."""

from .utils import DictProperty
from .base_dataset import TrajectoryDataset, random_split, DatasetModes, SPATIAL_COLUMNS, LAT, LON
from .fs_nyc import FSNYCDataset, FSNYC
from .geolife import GeoLifeDataset, GEOLIFE
from .padding import pad_feature_first
from .mnist_data import mnist_sequential, MNISTSequential
from .dataset_factory import get_dataset, Datasets

DATASET_CLASSES = {
    FSNYC: FSNYCDataset,
    GEOLIFE: GeoLifeDataset,
    MNISTSequential: mnist_sequential
}
