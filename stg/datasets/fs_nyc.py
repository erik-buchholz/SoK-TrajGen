#!/usr/bin/env python3
"""Provides access to the Foursquare NYC dataset as PyTorch dataset."""
import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from stg import config
from stg.datasets.base_dataset import TrajectoryDataset, DatasetModes, SPATIAL_COLUMNS, SPATIAL_FEATURE
from stg.utils.data import normalize_points

# Dataset Name
FSNYC = 'fs'

# Paths
PATH_ALL = config.BASE_DIR + 'data/fs_nyc/all_latlon.csv'
PATH_TRAIN = config.BASE_DIR + 'data/fs_nyc/train_latlon.csv'
PATH_TEST = config.BASE_DIR + 'data/fs_nyc/test_latlon.csv'

# Column Names
TID = 'tid'
UID = 'label'
HOUR = 'hour'
DAY = 'day'
CAT = 'category'

# Constants
MAX_LEN = 144

log = logging.getLogger()


class FSNYCDataset(TrajectoryDataset):

    def __init__(
            self,
            mode: DatasetModes = DatasetModes.ALL,
            spatial_columns: list = SPATIAL_COLUMNS,
            latlon_only: bool = True,
            normalize: bool = True,
            return_labels: bool = False,
            sort: bool = False,
            padding: bool = False,
            path: str = None,
            keep_original: bool = False,
            reference_point: List[float] or dict = None,
            scale_factor: List[float] or dict = None,
    ):
        """
        Initialize the dataset.
        :param mode: Dataset mode
        :param path: Absolute path to the dataset file(s). Either csv file or directory containing files.
        :param spatial_columns: List of columns to read in order
        :param latlon_only: Only consider spatial information
        :param normalize: Normalize values to [-1;1]
        :param return_labels: Return TIDs as labels
        :param sort: Order the encodings in increasing order of sequence length to reduce required padding
        :param padding:
        :param keep_original: Keep the original data in memory for testing
        :param reference_point: Reference point(s) for normalization (in order of spatial_columns)
        :param scale_factor: Scale factor(s) for normalization (in order of spatial_columns)
        """
        # Call superclass
        super().__init__(name=FSNYC, spatial_columns=spatial_columns, mode=mode, latlon_only=latlon_only, sort=sort,
                         return_labels=return_labels, normalize=normalize, path=path, reference_point=reference_point,
                         scale_factor=scale_factor, max_len=MAX_LEN)
        self.padding = padding
        self.keep_original = keep_original

        if self.mode == DatasetModes.PATH:
            self.path = path
        elif self.mode == DatasetModes.ALL:
            self.path = PATH_ALL
        elif self.mode == DatasetModes.TRAIN:
            self.path = PATH_TRAIN
        elif self.mode == DatasetModes.TEST:
            self.path = PATH_TEST
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

        # Read data
        log.info(f"Reading trajectories from '{self.path}'.")
        df = pd.read_csv(self.path, dtype={TID: str, UID: 'int32', HOUR: 'int32', DAY: 'int32', CAT: 'int32'})

        # Verify correct format
        num_cat = df[CAT].nunique() if CAT in df else 0
        assert HOUR not in df or df[HOUR].nunique() <= 24, "More than 24 values for Hour of Day"
        assert DAY not in df or df[DAY].nunique() <= 7, "More than 7 values for Day of Week"

        if self.keep_original:
            for tid, t in df.groupby(TID):
                self.originals[tid] = t
        if normalize:
            df, self.reference_point, self.scale_factor = normalize_points(
                df,
                columns=spatial_columns,
                ref=self.reference_point,
                scale_factor=self.scale_factor
            )

        self.features = [SPATIAL_FEATURE, ]
        if HOUR in df:
            self.features.append(HOUR)
        if DAY in df:
            self.features.append(DAY)
        if CAT in df:
            self.features.append(CAT)

        for tid, df in df.groupby(TID):
            self.tids.append(tid)
            self.uids[tid] = df[UID].iloc[0]
            lat_lon = torch.from_numpy(df[spatial_columns].to_numpy())
            if latlon_only:
                encoding = lat_lon
            else:
                encoding = [lat_lon, ]
                if HOUR in df:
                    encoding.append(torch.from_numpy(np.eye(24)[df[HOUR].to_numpy()]))
                if DAY in df:
                    encoding.append(torch.from_numpy(np.eye(7)[df[DAY].to_numpy()]))
                if CAT in df:
                    encoding.append(torch.from_numpy(np.eye(num_cat)[df[CAT].to_numpy()]))
            self.encodings[tid] = encoding

        if sort:
            self.tids = sorted(self.tids, key=lambda x: len(self.encodings[x]))

        if padding:
            self.lengths = {tid: len(self.encodings[tid]) for tid in self.tids}
            encodings = [self.encodings[tid] for tid in self.tids]
            encodings = pad_sequence(encodings, batch_first=True, padding_value=0.0)
            self.encodings = {tid: encodings[i] for i, tid in enumerate(self.tids)}

        assert len(self.tids) == len(self.encodings), f"{len(self.tids)} != {len(self.encodings)}"

    def __getitem__(self, index):
        tid = self.tids[index]
        output = [self.encodings[tid], ]
        if self.padding:
            output.append(self.lengths[tid])
        if self.return_labels:
            output.append(tid)
        return output if len(output) > 1 else output[0]


if __name__ == "__main__":
    # Create object
    ds = FSNYCDataset()
    assert len(ds) == 3079, f"Dataset contains {len(ds)} trajectories but expected 3,079."
    print("Successfully created dataset object")
