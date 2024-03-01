#!/usr/bin/env python3
"""PyTorch Wrapper for GeoLife Dataset by Microsoft.
URL: [Microsoft Research](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
"""
import argparse
import logging
from collections import OrderedDict
import multiprocessing as mp
from enum import Enum
from typing import List, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from tqdm import tqdm

from stg.datasets import DictProperty, TrajectoryDataset, DatasetModes, SPATIAL_COLUMNS
from stg import config
from stg.datasets import preprocessing as pp
from stg.datasets.base_dataset import SPATIAL_FEATURE
from stg.utils.data import normalize_points
from stg.utils.helpers import find_bbox, get_ref_point, get_scaling_factor

# Dataset Name
GEOLIFE = 'geolife'
GEOLIFE_DIR = config.BASE_DIR + 'data/geolife/Data/'
NUM_FILES = 18670  # Determined via `lsl data/geolife/Data/???/Trajectory/ | grep '.plt' | wc -l`
NUM_PROCESSED_TRAJS = 69504  # See geoLife.ipynb
SAMPLING_RATE = 5  # [s]
MAXIMAL_BREAK_INTERVAL = 60  # [s]
MIN_LEN = 10
MAX_LEN = 200
# BBOX_QUANTILE = 0.95
GEOLIFE_DATATYPES = OrderedDict({
    "latitude": float,
    "longitude": float,
    "ignore": str,
    "altitude": float,
    "days": float,
    "date": str,
    "time": str,
})
LAT = 'lat'
LON = 'lon'
log = logging.getLogger()
# Reference Point and Scale Factor
# Due to dataset size, on-demand computation is very expensive
# 90-Percentile BBOX
GL_REF_POINT_90 = None
GL_SCALE_FACTOR_90 = None
# 95-Percentile BBOX
GL_REF_POINT_95 = {LAT: 35.60931733333335,
                   LON: 113.2185733333335}
GL_SCALE_FACTOR_95 = {LAT: 4.76171566666665,
                      LON: 4.748456666666499}
GL_REF_POINT_FIFTH = {LAT: 39.89000165,
                      LON: 116.37499995}
GL_SCALE_FACTOR_FIFTH = {LAT: 0.13999835,
                         LON: 0.18499995}

DEFAULT_DATASET_PATH = config.BASE_DIR + "data/geolife_FIFTH-RING_5_60_200_TRUNCATE"
DEFAULT_REF_POINT = GL_REF_POINT_FIFTH
DEFAULT_SCALE_FACTOR = GL_SCALE_FACTOR_FIFTH
DEFAULT_BBOX = "FIFTH-RING"

GL_BBOXES = {
    90: (115.65822549999999, 116.688371, 36.3831297, 40.072966),
    95: (108.4700764500001, 117.967031749, 30.847590175850016, 40.371033),
    'FIFTH-RING': (116.19, 116.56, 39.75, 40.03),  # Fifth ring road of Beijing
    'SIXTH-RING': (116.07, 116.72, 39.68, 40.18),  # Sixth ring road of Beijing
}


class TrajectoryLength(Enum):
    REMOVE = 1
    TRUNCATE = 2
    SPLIT = 3


LENGTH_METHOD = TrajectoryLength.TRUNCATE


def get_filename(bbox_name: str, sampling_rate: int, maximal_break_interval: int, max_len: int,
                 length_method: TrajectoryLength):
    return f"geolife_{bbox_name}_{sampling_rate}_{maximal_break_interval}_{max_len}_{length_method.name}"


class GeoLifeDataset(TrajectoryDataset):
    """
    Pre-processed GeoLife dataset
    """

    def __init__(self,
                 mode: DatasetModes = DatasetModes.PATH,
                 path: str = None,
                 spatial_columns: list = SPATIAL_COLUMNS,
                 latlon_only: bool = True,
                 normalize: bool = True,
                 return_labels: bool = False,
                 sort: bool = False,
                 keep_original: bool = False,
                 as_dataframe: bool = False,
                 reference_point: List[float] or dict = DEFAULT_REF_POINT,
                 scale_factor: List[float] or dict = DEFAULT_SCALE_FACTOR,
                 ):
        # Validate inputs
        if sort:
            raise NotImplementedError("Sorting is not implemented due to size of GeoLife dataset.")

        # Call superclass / store parameters
        features = [SPATIAL_FEATURE] if latlon_only else [SPATIAL_FEATURE, 'hour', 'day']
        super().__init__(name=GEOLIFE, spatial_columns=spatial_columns, features=features, mode=mode,
                         latlon_only=latlon_only,
                         sort=sort, return_labels=return_labels, normalize=normalize, path=path,
                         keep_original=keep_original, reference_point=reference_point, scale_factor=scale_factor,
                         max_len=MAX_LEN)
        self.as_dataframe = as_dataframe

        log.warning("Make sure the reference point and scale factor computed during pre-processing are used!")

        if self.mode == DatasetModes.ALL:
            self.path = DEFAULT_DATASET_PATH
        elif self.mode != DatasetModes.PATH:
            raise NotImplementedError("Only path mode is supported at the moment.")

        log.info(f"Reading trajectories from '{self.path}'.")
        path = Path(self.path)
        for child in path.glob("*.csv"):
            tid = child.name.replace('.csv', '')
            self.tids.append(tid)

        self.tids = sorted(self.tids)

        # Original DataFrames are generated on demand --> replace dict by property
        self._originals = {}
        self.originals = DictProperty(self._originals, self.tids, self._encode)

        if NUM_PROCESSED_TRAJS is not None:
            assert len(self.tids) == NUM_PROCESSED_TRAJS, (f"Found {len(self.tids)} files, "
                                                           f"expected {NUM_PROCESSED_TRAJS}.")

    def _encode(self, tid: str) -> None:
        # Load from file
        df = pd.read_csv(
            f"{self.path}/{tid}.csv",
            dtype={'tid': str,
                   'uid': int,
                   LAT: float,
                   LON: float,
                   'altitude': float
                   },
            parse_dates=['timestamp']
        )
        if self.keep_original:
            self._originals[tid] = df
        self.uids[tid] = df['uid'][0]
        if self.as_dataframe:
            self.encodings[tid] = df
        else:
            # Convert to Tensor
            latlon = df[self.columns]
            if self.normalize:
                latlon, _, _ = normalize_points(latlon, ref=self.reference_point, scale_factor=self.scale_factor,
                                                columns=self.columns)
            latlon = torch.from_numpy(latlon.to_numpy())
            if self.latlon_only:
                self.encodings[tid] = latlon
            else:
                # Transform timestamp into one hot encoding for hour of day and day of week
                hour = torch.from_numpy(np.eye(24)[df['timestamp'].dt.hour.to_numpy()])
                day = torch.from_numpy(np.eye(7)[df['timestamp'].dt.dayofweek.to_numpy()])
                self.encodings[tid] = [latlon, hour, day]

    def __getitem__(self, index: int):
        tid = self.tids[index]
        if tid not in self.encodings:
            # Generate encoding if not existing
            self._encode(tid)

        if self.return_labels:
            return self.encodings[tid], tid
        else:
            return self.encodings[tid]


def _read_geolife_file(root_dir: str, uid: int, tid: str) -> pd.DataFrame:
    """Read the original GeoLife files into dataFrame.
    :param root_dir: Root directory of the GeoLife dataset
    :param uid: User ID
    :param tid: Trajectory ID
    :return: DataFrame containing the trajectory
    """
    filename = f"{root_dir}/{uid:03d}/Trajectory/{tid}.plt"
    dates = [['date', 'time']]
    df = pd.read_csv(
        filename,
        delimiter=',',
        header=None,
        names=list(GEOLIFE_DATATYPES.keys()),
        skiprows=6,
        dtype=GEOLIFE_DATATYPES,
        parse_dates=dates
    )
    df.rename(columns={'date_time': 'timestamp', 'latitude': LAT, 'longitude': LON}, inplace=True)
    df.drop(inplace=True, columns=['ignore', 'days'])
    df['uid'] = uid
    df['tid'] = f"{uid:03d}_{tid}"
    return df


class GeoLifeUnprocessed(Dataset):
    """Unprocessed GeoLife dataset"""

    def __init__(self, path: str = GEOLIFE_DIR):
        self.root = path

        self.uids = []
        self.tids = []

        root_dir = Path(self.root)

        # Check if Path exists and raise error if not
        if not root_dir.exists():
            raise ValueError(f"GeoLife dataset not found at {root_dir}."
                             f"Make sure you downloaded it into {GEOLIFE_DIR} or you specified the correct path.")

        for child in root_dir.iterdir():
            if child.is_dir():
                uid = int(child.name)
                self.uids.append(uid)
                for traj in child.glob("Trajectory/*.plt"):
                    t_name = f"{uid:03d}_{traj.name.replace('.plt', '')}"
                    self.tids.append(t_name)

        self.uids = sorted(self.uids)
        self.tids = sorted(self.tids)

        assert len(self.tids) == NUM_FILES, (f"Found {len(self.tids)} files, expected {NUM_FILES}."
                                             f"Downloaded dataset may be incomplete.")

    def __len__(self):
        return len(self.tids)

    def __getitem__(self, index: int):
        tid = self.tids[index]
        uid, tid = tid.split('_')
        uid = int(uid)
        return _read_geolife_file(self.root, uid, tid)


def wrap_resample(t):
    return pp.resample_trajectory(*t)


def preprocess_geolife(
        input_dir: str = GEOLIFE_DIR,
        sampling_rate: int = SAMPLING_RATE,
        split_interval: float = MAXIMAL_BREAK_INTERVAL,
        min_length: int = MIN_LEN,
        max_length: int = MAX_LEN,
        length_method: TrajectoryLength = LENGTH_METHOD,
        bbox_quantile: float or None = None,
        bbox: (float, float, float, float) = None,
        bbox_name: str = None,
        overwrite: bool = False,
        test: bool = False
):
    """
    Preprocess GeoLife dataset.
    :param input_dir: Directory containing the unzipped GeoLife dataset
    :param sampling_rate: Sampling rate in seconds
    :param split_interval: Maximum break interval in seconds
    :param min_length: Minimum length of any trajectory
    :param max_length: Maximum length of any trajectory
    :param length_method: How to handle trajectories that are too long
    :param bbox_quantile: Bounding Box quantile. OR
    :param bbox: Bounding Box (instead of quantile): (min_lon, max_lon, min_lat, max_lat) OR
    :param bbox_name: Name of the BBOX (for filename)
    :param overwrite: Overwrite existing files
    :param test: Run in test mode (don't save)
    :return:
    """
    # Validate that exactly one of the three bbox values is defined
    bbox_parameters = [bbox_quantile, bbox, bbox_name]
    non_none_parameters = sum([1 for param in bbox_parameters if param is not None])
    if non_none_parameters != 1:
        # raise ValueError("Exactly one of bbox_quantile, bbox, or bbox_name should be provided.")
        log.warning("Exactly one of bbox_quantile, bbox, or bbox_name should be provided.")
        log.warning(f"Using default bbox: {DEFAULT_BBOX}.")
        bbox_name = DEFAULT_BBOX
    if bbox_quantile is not None:
        bbox_name = f"{bbox_quantile * 100:.0f}-quantile"
    elif bbox is not None:
        bbox_name = f"CUSTOM"

    # Make sure output dir does not exist and create it
    output_dir = config.BASE_DIR + f"data/{get_filename(bbox_name, sampling_rate, split_interval, max_length, length_method)}"
    out_dir = Path(output_dir)
    if out_dir.exists() and not overwrite and not test:
        raise ValueError("Output directory exists.")
    else:
        print(f"Writing to {out_dir}.")
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load unprocessed dataset
    gl_all = GeoLifeUnprocessed(path=input_dir)

    # Make sure the full dataset has been loaded and no files are missing
    assert len(gl_all) == NUM_FILES
    print(f"Unprocessed GeoLife Trajectories:\t\t{len(gl_all)}")
    geolife_all = pd.concat(gl_all)
    n_unprocessed = len(geolife_all)
    print(f"Number of points in GeoLife Dataset:\t\t{n_unprocessed:,}")

    # Compute bbox
    if bbox is None:
        if bbox_quantile is not None:
            bbox = find_bbox(geolife_all, bbox_quantile)
        elif bbox_name in GL_BBOXES:
            bbox = GL_BBOXES[bbox_name]
        else:
            raise ValueError(f"Unknown BBOX name {bbox_name}.")
    print(f"Bounding Box: {bbox_name} =\t\t\t{bbox}")

    # Drop out of bounds points
    tasks = ((t, *bbox, LAT, LON, False) for t in gl_all)
    with mp.Pool() as pool:
        geolife_in_bbox = pool.starmap(pp.drop_out_of_bounds,
                                       tqdm(tasks, total=len(gl_all), desc="Drop out-of-bounds"))
    n_in_bbox = len(pd.concat(geolife_in_bbox))
    print(f"Number of Points after dropping out of BBOX:\t{n_in_bbox:,} ({n_in_bbox / n_unprocessed * 100:.2f}%)")

    # Down-sampling
    chunksize = 100  # See geolife.ipynb
    tasks = ((t, sampling_rate) for t in geolife_in_bbox)

    # Multiprocessing code with tqdm progress bar
    geolife_resampled = []
    with mp.Pool() as pool:
        with tqdm(total=len(geolife_in_bbox), desc="Resampling") as pbar:
            for t in pool.imap_unordered(wrap_resample, tasks, chunksize=chunksize):
                if len(t) > 0:
                    geolife_resampled.append(t)
                pbar.update()

    geolife_resampled_df = pd.concat(geolife_resampled)
    n_resampled = len(geolife_resampled_df)
    print(
        f"Number of Points after resampling to {sampling_rate}s:\t{n_resampled:,} "
        f"({n_resampled / n_unprocessed * 100:.2f}%)")
    print(
        f"Number of Trajectories after resampling to {sampling_rate}s:\t{len(geolife_resampled):,} "
        f"({len(geolife_resampled) / len(gl_all) * 100:.2f}%)")

    # Split trajectories on breaks
    geolife_split = []
    for i, t in tqdm(enumerate(geolife_resampled), total=len(geolife_resampled), desc="Splitting trajectories"):
        splits = pp.split_and_rename(t, interval=split_interval, max_len=None)
        geolife_split.extend(splits)

    geolife_split_df = pd.concat(geolife_split)
    n_resampled_split = len(geolife_split_df)
    print(
        f"Number of Points after splitting at {split_interval}s:\t{n_resampled_split:,} "
        f"({n_resampled_split / n_unprocessed * 100:.2f}%)")
    print(
        f"Number of Trajectories after splitting at {split_interval}s:\t{len(geolife_split):,} "
        f"({len(geolife_split) / len(gl_all) * 100:.2f}%)")

    # Upper Length
    pb = tqdm(geolife_split, total=len(geolife_split), desc="Truncating trajectories")

    if length_method == TrajectoryLength.REMOVE:
        # Remove trajectories that are longer than 200 points
        geolife_upper_bounded = [t.reset_index(drop=True) for t in pb if len(t) <= MAX_LEN]
    elif length_method == TrajectoryLength.TRUNCATE:
        # Truncate trajectories to MAX_LEN points
        geolife_upper_bounded = [t[:MAX_LEN].reset_index(drop=True) for t in pb]
    elif length_method == TrajectoryLength.SPLIT:
        # Split trajectories into multiple trajectories of MAX_LEN points or fewer
        geolife_upper_bounded = []
        for t in pb:
            splits = pp.split_and_rename(t, interval=None, max_len=MAX_LEN)
            geolife_upper_bounded.extend(splits)
    else:
        raise ValueError(f"Invalid Trajectory Length Method: {length_method}")

    geolife_upper_bounded_df = pd.concat(geolife_upper_bounded)
    n_upper_bounded = len(geolife_upper_bounded_df)
    print(
        f"Number of Points after upper bounding (via {length_method.name}) to {max_length} points:\t\t"
        f"{n_upper_bounded:,} ({n_upper_bounded / n_unprocessed * 100:.2f}%)")
    print(
        f"Number of Trajectories after upper bounding (via {length_method.name}) to {max_length} points:"
        f"\t{len(geolife_upper_bounded):,} ({len(geolife_upper_bounded) / len(gl_all) * 100:.2f}%)")

    # Remove too short trajectories
    geolife_final = [t for t in geolife_upper_bounded if len(t) >= min_length]

    geolife_final_df = pd.concat(geolife_final)
    n_final = len(geolife_final_df)
    print(f"Final Number of Points:\t\t{n_final:,} ({n_final / n_unprocessed * 100:.2f}%)")
    print(
        f"Final Number of Trajectories:\t{len(geolife_final):,} "
        f"({len(geolife_final) / len(gl_all) * 100:.2f}%)")

    # Sort by uid first and tid second
    geolife_final.sort(key=lambda x: (x['uid'].iloc[0], x['tid'].iloc[0]))

    if not test:
        # Store the dataframe
        geolife_final_df.to_csv(output_dir + '.csv', index=False)
        # Store into separate files for each trajectory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i, t in tqdm(enumerate(geolife_final), total=len(geolife_final), desc="Storing Trajectories"):
            tid = f"{t['tid'].iloc[0]}"
            t.to_csv(output_dir + f"/{tid}.csv", index=False)

        print(f"Saved to {output_dir}.")

    print(f"Done. Generated {len(geolife_final):,} GeoLife trajectories.")

    # Determine reference point
    ref_point = get_ref_point(geolife_final_df[['lat', 'lon']])
    print(f"Reference Point:\t{ref_point}")

    # Determine scaling factor
    scaling_factor = get_scaling_factor(geolife_final_df[['lat', 'lon']], ref_point)
    print(f"Scaling Factor:\t\t{scaling_factor}")

    return geolife_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--directory', type=str, default=PROCESSED_DIR)
    parser.add_argument('--split_interval', type=float, default=MAXIMAL_BREAK_INTERVAL)
    parser.add_argument('--sampling_rate', type=int, default=SAMPLING_RATE)
    parser.add_argument('--length_method', type=TrajectoryLength, choices=list(TrajectoryLength),
                        default=LENGTH_METHOD)
    parser.add_argument('--max_len', type=int, default=MAX_LEN)
    parser.add_argument('--min_len', type=int, default=MIN_LEN)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--test', action='store_true')
    # Add argument group for special BBOXes
    bbox_group = parser.add_mutually_exclusive_group()
    bbox_group.add_argument('-b', '--bbox_quantile', type=float, default=None,
                            help="Bounding Box quantile. All points outside the quantile are dropped.")
    # Add argument for fifth ring
    bbox_group.add_argument('--fifth', action='store_true',
                            help="Use the fifth ring of Beijing as BBOX.")
    # Add argument for sixth ring
    bbox_group.add_argument('--sixth', action='store_true',
                            help="Use the sixth ring of Beijing as BBOX.")
    opt = parser.parse_args()
    if opt.fifth:
        bbox_name = 'FIFTH-RING'
    elif opt.sixth:
        bbox_name = 'SIXTH-RING'
    else:
        bbox_name = None
    print("Arguments: ", opt)

    preprocess_geolife(
        split_interval=opt.split_interval,
        sampling_rate=opt.sampling_rate,
        min_length=opt.min_len,
        max_length=opt.max_len,
        length_method=opt.length_method,
        bbox_quantile=opt.bbox_quantile,
        bbox_name=bbox_name,
        overwrite=opt.overwrite,
        test=opt.test
    )
