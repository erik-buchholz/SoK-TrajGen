#!/usr/bin/env python3
"""Base Dataset Class for Interface Definition"""
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Sequence, Dict, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

# Constants
LAT = 'lat'
LON = 'lon'
SPATIAL_COLUMNS = [LON, LAT]
SPATIAL_FEATURE = 'latlon'


class DatasetModes(str, Enum):
    """Enumeration of dataset modes."""
    TRAIN = 'train'
    TEST = 'test'
    ALL = 'all'
    PATH = 'path'



class TrajectoryDataset(Dataset, ABC):

    @abstractmethod
    def __init__(self, spatial_columns: list, name: str, max_len: int, mode: DatasetModes = DatasetModes.ALL,
                 features=[SPATIAL_FEATURE], latlon_only: bool = True, sort: bool = False, return_labels: bool = True,
                 normalize: bool = True, path: str = None, keep_original: bool = False,
                 reference_point: List[float] or dict = None, scale_factor: List[float] or dict = None, **kwargs):
        """
        Initialize the dataset.
        :param mode: Dataset mode
        :param name: Name of the dataset
        :param max_len: Maximum length of a trajectory (required by some models)
        :param path: Absolute path to the dataset file(s). Either csv file or directory containing files.
                     Overwrites Mode.
        :param spatial_columns: List of lat/lon(/alt) or x/y/z columns to read in order
        :param features: List of features to use; spatial features are always called 'latlon'.
        :param latlon_only: Only consider spatial information
        :param normalize: Normalize values to [-1;1]
        :param return_labels: Return TIDs as labels
        :param sort: Order the encodings in increasing order of sequence length to reduce required padding
        :param keep_original: Keep the original data in memory for testing
        :param reference_point: Reference point(s) for normalization. Required if normalize=True.
        :param scale_factor: Scale factor(s) for normalization. Required if normalize=True.
        """
        super().__init__()
        # Store parameters
        self.name = name
        self.mode = DatasetModes.PATH if path is not None else mode
        self.path = path
        self.columns = spatial_columns
        self.features = features
        self.latlon_only = latlon_only
        # Raise value error if latlon_only true and features contains non-spatial features
        if self.latlon_only and any(f not in [SPATIAL_FEATURE] for f in self.features):
            raise ValueError("Cannot use non-spatial features if latlon_only is True.")
        self.normalize = normalize
        self.return_labels = return_labels
        self.keep_original = keep_original
        self.max_len = max_len
        self.sort = sort
        if type(reference_point) is dict:
            self.reference_point = [reference_point[c] for c in self.columns]
        else:
            self.reference_point = reference_point
        if type(scale_factor) is dict:
            self.scale_factor = [scale_factor[c] for c in self.columns]
        else:
            self.scale_factor = scale_factor

        # Initialize Lookup Tables
        self.tids: List[str] = []  # List of trajectory IDs
        self.uids: Dict[str, Union[int, str]] = {}  # TID --> UID
        self.encodings: Dict[str, torch.Tensor] = {}  # TID --> Encoding
        self.originals: Dict[str, pd.DataFrame] = {}  # TID --> Original Dataframe

    def __len__(self):
        """Return the number of trajectories in the dataset."""
        return len(self.tids)

    @abstractmethod
    def __getitem__(self, index):
        """Return the trajectory at the given index."""
        # Must be implemented by subclass
        pass


class SubTrajectoryDataset(TrajectoryDataset):
    dataset: TrajectoryDataset
    indices: Sequence[int]
    tids: List[str]

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.tids = [dataset.tids[idx] for idx in indices]
        self.uids: Dict[str, Union[int, str]] = ProxyAttribute(self, 'uids')
        self.encodings: Dict[str, torch.Tensor] = ProxyAttribute(self, 'encodings')
        self.originals: Dict[str, pd.DataFrame] = ProxyAttribute(self, 'originals')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    # Pass through all other methods except the defined ones, uid, encoding and original
    def __getattr__(self, item):
        """Return the attribute from the dataset. For uid, encoding and original, only return value if accessed index is in self.tids or raise ValueError."""
        return getattr(self.dataset, item)


# Create Proxy for uid, encoding and original that verifies that the index is in self.tids
class ProxyAttribute:
    def __init__(self, dataset: SubTrajectoryDataset, attr: str) -> None:
        self.parent = dataset
        self.attr = attr

    def __getitem__(self, tid):
        if tid not in self.parent.tids:
            raise ValueError(f"Trajectory ID '{tid}' not in dataset.")
        return self.parent.dataset.__getattribute__(self.attr)[tid]

    def values(self):
        return [self.parent.dataset.__getattribute__(self.attr)[tid] for tid in self.parent.tids]


def random_split(dataset: TrajectoryDataset, sizes: List[Union[float, int]]) -> List[TrajectoryDataset]:
    """
    Split the dataset based on provided sizes or fractions.

    :param dataset: The dataset to split.
    :param sizes: A list of dataset sizes (as integers) or fractions (as floats).
    Returns:
        A list of SubTrajectoryDataset instances.
    """
    all_indices = list(range(len(dataset)))

    # Determine if sizes represent absolute lengths or fractions
    if all(isinstance(size, int) for size in sizes):
        assert sum(sizes) == len(dataset), "The sum of sizes should be equal to the total dataset size."
        split_sizes = sizes
    elif all(isinstance(size, float) for size in sizes):
        assert abs(sum(sizes) - 1.0) < 1e-6, "The sum of fractions should be approximately 1."
        split_sizes = [int(len(dataset) * size) for size in sizes[:-1]]  # everything except the last size
        split_sizes.append(len(dataset) - sum(split_sizes))  # remaining data for the last size
    else:
        raise ValueError("Sizes should either be all integers or all floats.")

    subsets = []
    for split_size in split_sizes:
        split_indices = random.sample(all_indices, split_size)
        all_indices = [i for i in all_indices if i not in split_indices]
        subsets.append(SubTrajectoryDataset(dataset, split_indices))

    return subsets
