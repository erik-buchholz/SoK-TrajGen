#!/usr/bin/env python3
"""Test the GeoLife Dataset."""
import logging
import os
import random
import unittest
from pathlib import Path
from unittest import skipUnless
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import torch

from stg.datasets.base_dataset import DatasetModes, LAT, LON, SPATIAL_COLUMNS
from stg.datasets.geolife import GeoLifeDataset, GeoLifeUnprocessed, DEFAULT_REF_POINT, DEFAULT_SCALE_FACTOR, \
    GEOLIFE_DIR, DEFAULT_DATASET_PATH as GL_PROCESSED_DIR
from stg.utils import helpers
from stg.utils.data import normalize_points

log = logging.getLogger()


@skipUnless(Path(GEOLIFE_DIR).exists() or os.environ.get('FORCE_TESTS') == 'true',
            "GeoLife dataset not found. Ignoring GeoLife tests.")
class TestGeoLifeRaw(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Load raw dataset
        cls.gl = GeoLifeUnprocessed()

    def test_trajectory_num(self):
        """Test number of trajectories."""
        self.assertEqual(len(self.gl), 18670, "Wrong number of trajectories."
                                              "Verify that dataset was downloaded correctly.")

    @skipUnless(os.environ.get('RUN_SLOW_TESTS') == 'true', "Concatenating all points takes ~2min")
    def test_point_num(self):
        """Test number of points."""
        self.assertEqual(sum(len(t) for t in self.gl), 24876978, "Wrong number of points:"
                                                                 "Verify that dataset was downloaded correctly.")


@skipUnless(Path(GL_PROCESSED_DIR).exists() or os.environ.get('FORCE_TESTS') == 'true',
            "GeoLife dataset not found. Ignoring GeoLife tests.")
@skipUnless(os.environ.get('RUN_SLOW_TESTS') == 'true', "Concatenating all points takes >2min")
class TestGeoLifeProcessedAll(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Load pre-processed dataset
        cls.gl = GeoLifeDataset(
            mode=DatasetModes.ALL,
            normalize=True,
            return_labels=False,
            as_dataframe=True
        )
        cls.all_points = pd.concat(t for t in cls.gl)

    def test_point_num(self):
        """Test number of points."""
        self.assertEqual(len(self.all_points), 6486421, "Wrong number of points.")

    def test_check_reference_point(self):
        """Check reference point."""
        columns = SPATIAL_COLUMNS
        ref = helpers.get_ref_point(self.all_points[columns])
        self.assertEqual(len(ref), 2, "Reference point has wrong dimension.")
        if type(DEFAULT_REF_POINT) is dict:
            expected = [DEFAULT_REF_POINT[c] for c in columns]
        else:
            expected = DEFAULT_REF_POINT
        self.assertTrue(np.allclose(ref, expected, atol=1e-5),
                        "Reference point is wrong.")

    def test_check_scale(self):
        ref = [DEFAULT_REF_POINT[c] for c in SPATIAL_COLUMNS] if type(DEFAULT_REF_POINT) is dict else DEFAULT_REF_POINT
        sf = helpers.get_scaling_factor(self.all_points[SPATIAL_COLUMNS], ref)
        expected_sf = [DEFAULT_SCALE_FACTOR[c] for c in SPATIAL_COLUMNS] if type(
            DEFAULT_SCALE_FACTOR) is dict else DEFAULT_SCALE_FACTOR
        self.assertTrue(np.allclose(sf, expected_sf, atol=1e-5),
                        "Scale factor is wrong.")


@skipUnless(Path(GL_PROCESSED_DIR).exists() or os.environ.get('FORCE_TESTS') == 'true',
            "GeoLife dataset not found. Ignoring GeoLife tests.")
class TestGeoLifeProcessedQuick(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Disable logging
        logging.disable(logging.CRITICAL)
        cls.gl = GeoLifeDataset(
            mode=DatasetModes.ALL,
            normalize=True,
            return_labels=False,
            as_dataframe=True
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # Re-enable logging
        logging.disable(logging.NOTSET)

    @patch.object(Path, "glob")
    def setUp(self, mock_glob):
        # Mock the behavior of path.glob("*.csv")
        mock_file = MagicMock()
        mock_file.name = "1.csv"
        mock_glob.return_value = [mock_file] * 69504  # mock files with the name "1.csv"
        self.dataset = GeoLifeDataset(
            mode=DatasetModes.ALL,
            normalize=True,
            return_labels=False
        )
        self.dataset.scale_factor = (1.0, 1.0)  # Easier testing

    def test_sort(self):
        """Test sort parameters raises an error."""
        with self.assertRaises(NotImplementedError):
            GeoLifeDataset(
                mode=DatasetModes.ALL,
                normalize=True,
                return_labels=False,
                sort=True
            )

    def test_trajectory_num(self):
        """Test number of trajectories."""
        self.assertEqual(len(self.gl), 69504, "Wrong number of trajectories.")

    @patch.object(Path, "glob")
    @patch("pandas.read_csv")
    def test_encoding_latlon_only(self, mock_read_csv, mock_glob):
        # Mocked DataFrame
        data = {
            'tid': ['1'],
            'uid': [1],
            LAT: [40.0],
            LON: [116.3],
            'altitude': [100.0],
            'timestamp': pd.to_datetime(['2022-01-01 08:00:00'])
        }
        mock_read_csv.return_value = pd.DataFrame(data)

        # Mock the behavior of path.glob("*.csv")
        mock_file = MagicMock()
        mock_file.name = "1.csv"
        mock_glob.return_value = [mock_file] * 69504  # mock files with the name "1.csv"

        gl = GeoLifeDataset(
            mode=DatasetModes.ALL,
            normalize=True,
            return_labels=True,
            as_dataframe=False,
            keep_original=True,
            latlon_only=True
        )

        for _ in range(10):
            encoding, tid = random.choice(gl)
            original = gl.originals[tid]

            encoded_original, _, _ = normalize_points(original, ref=gl.reference_point,
                                                      scale_factor=gl.scale_factor,
                                                      columns=gl.columns, inplace=False)
            encoded_original = torch.from_numpy(encoded_original[gl.columns].to_numpy())
            self.assertTrue(torch.allclose(encoding, encoded_original, atol=1e-5),
                            "Encoding is wrong for TID {}".format(tid))

    @patch.object(Path, "glob")
    @patch("pandas.read_csv")
    def test_encoding_with_time_features(self, mock_read_csv, mock_glob):
        # Mocked DataFrame
        data = {
            'tid': ['1'],
            'uid': [1],
            LAT: [40.0],
            LON: [116.3],
            'altitude': [100.0],
            'timestamp': pd.to_datetime(['2022-01-01 08:00:00'])
        }
        mock_read_csv.return_value = pd.DataFrame(data)

        # Mock the behavior of path.glob("*.csv")
        mock_file = MagicMock()
        mock_file.name = "1.csv"
        mock_glob.return_value = [mock_file] * 69504  # mock files with the name "1.csv"

        gl = GeoLifeDataset(
            mode=DatasetModes.ALL,
            normalize=True,
            return_labels=True,
            as_dataframe=False,
            keep_original=True,
            latlon_only=False
        )

        for _ in range(10):
            # Reset mock at start of each round
            encoding, tid = random.choice(gl)
            original = gl.originals[tid]

            # Validate latlon encoding
            encoded_original, _, _ = normalize_points(original, ref=gl.reference_point,
                                                      scale_factor=gl.scale_factor,
                                                      columns=gl.columns, inplace=False)
            encoded_original = torch.from_numpy(encoded_original[gl.columns].to_numpy())
            self.assertTrue(torch.allclose(encoding[0], encoded_original, atol=1e-5),
                            "LatLon encoding is wrong for TID {}".format(tid))

            # Validate hour one-hot encoding
            original_hour = original['timestamp'].dt.hour.to_numpy()
            one_hot_hour = np.eye(24)[original_hour]
            self.assertTrue(torch.all(torch.from_numpy(one_hot_hour) == encoding[1]),
                            "Hour one-hot encoding is wrong for TID {}".format(tid))

            # Validate day one-hot encoding
            original_day = original['timestamp'].dt.dayofweek.to_numpy()
            one_hot_day = np.eye(7)[original_day]
            self.assertTrue(torch.all(torch.from_numpy(one_hot_day) == encoding[2]),
                            "Day one-hot encoding is wrong for TID {}".format(tid))

    @patch("pandas.read_csv")
    def test_encode_without_normalization_and_latlon_only(self, mock_read_csv):
        # Mocking a sample dataframe with a trajectory of length 5
        mock_df = pd.DataFrame({
            'tid': ['t1', 't1', 't1', 't1', 't1'],
            'uid': [1, 1, 1, 1, 1],
            LAT: [40.0, 40.1, 40.2, 40.3, 40.4],
            LON: [116.3, 116.4, 116.5, 116.6, 116.7],
            'altitude': [10.0, 11.0, 12.0, 13.0, 14.0],
            'timestamp': [
                pd.Timestamp('2022-01-01 12:00:00'),
                pd.Timestamp('2022-01-01 13:00:00'),
                pd.Timestamp('2022-01-01 14:00:00'),
                pd.Timestamp('2022-01-01 15:00:00'),
                pd.Timestamp('2022-01-01 16:00:00')
            ]
        })
        mock_read_csv.return_value = mock_df

        tid = "t1"
        self.dataset.normalize = False
        self.dataset.latlon_only = True

        self.dataset._encode(tid)

        expected_tensor = torch.tensor([
            [116.3, 40.0],
            [116.4, 40.1],
            [116.5, 40.2],
            [116.6, 40.3],
            [116.7, 40.4]
        ], dtype=torch.float64)

        self.assertTrue(torch.equal(self.dataset.encodings[tid], expected_tensor),
                        f"Resulting tensor {self.dataset.encodings[tid]} is not equal to expected tensor {expected_tensor}.")

    @patch("pandas.read_csv")
    def test_encode_with_normalization_and_latlon_only(self, mock_read_csv):
        # Mocking a sample dataframe
        mock_df = pd.DataFrame({
            'tid': ['t1', 't1', 't1', 't1', 't1'],
            'uid': [1, 1, 1, 1, 1],
            LAT: [40.0, 40.1, 40.2, 40.3, 40.4],
            LON: [116.3, 116.4, 116.5, 116.6, 116.7],
            'altitude': [10.0, 11.0, 12.0, 13.0, 14.0],
            'timestamp': [
                pd.Timestamp('2022-01-01 12:00:00'),
                pd.Timestamp('2022-01-01 13:00:00'),
                pd.Timestamp('2022-01-01 14:00:00'),
                pd.Timestamp('2022-01-01 15:00:00'),
                pd.Timestamp('2022-01-01 16:00:00')
            ]
        })
        mock_read_csv.return_value = mock_df

        tid = "t1"
        self.dataset.normalize = True
        self.dataset.latlon_only = True
        self.dataset.scale_factor = (1.0, 1.0)  # Setting the scale factor

        self.dataset._encode(tid)

        expected_tensor, _, _ = normalize_points(mock_df, columns=[LON, LAT],
                                                 ref=self.dataset.reference_point,
                                                 scale_factor=self.dataset.scale_factor)
        expected_tensor = torch.from_numpy(expected_tensor[[LON, LAT]].to_numpy())
        self.assertTrue(torch.equal(self.dataset.encodings[tid], expected_tensor))

    @patch("pandas.read_csv")
    def test_encode_without_latlon_only(self, mock_read_csv):
        # Mocking a sample dataframe
        mock_df = pd.DataFrame({
            'tid': ['t1', 't1', 't1', 't1', 't1'],
            'uid': [1, 1, 1, 1, 1],
            LAT: [40.0, 40.1, 40.2, 40.3, 40.4],
            LON: [116.3, 116.4, 116.5, 116.6, 116.7],
            'altitude': [10.0, 11.0, 12.0, 13.0, 14.0],
            'timestamp': [
                pd.Timestamp('2022-01-01 12:00:00'),
                pd.Timestamp('2022-01-01 13:00:00'),
                pd.Timestamp('2022-01-01 14:00:00'),
                pd.Timestamp('2022-01-01 15:00:00'),
                pd.Timestamp('2022-01-01 16:00:00')
            ]
        })
        mock_read_csv.return_value = mock_df

        tid = "t1"
        self.dataset.latlon_only = False
        self.dataset.scale_factor = (1.0, 1.0)

        self.dataset._encode(tid)

        expected_latlon, _, _ = normalize_points(mock_df, columns=self.dataset.columns,
                                                 ref=self.dataset.reference_point,
                                                 scale_factor=self.dataset.scale_factor)
        expected_latlon = torch.from_numpy(expected_latlon[self.dataset.columns].to_numpy())
        # Hours from 12 to 16
        expected_hour_list = [
            [0.0] * 12 + [1.0] + [0.0] * 11,  # 12th hour
            [0.0] * 13 + [1.0] + [0.0] * 10,  # 13th hour
            [0.0] * 14 + [1.0] + [0.0] * 9,  # 14th hour
            [0.0] * 15 + [1.0] + [0.0] * 8,  # 15th hour
            [0.0] * 16 + [1.0] + [0.0] * 7  # 16th hour
        ]
        expected_hour = torch.tensor(expected_hour_list, dtype=torch.float64)
        # The 1st of January is a Saturday
        expected_day = torch.tensor(
            [[0.0] * 5 + [1.0] + [0.0]] * 5,
            dtype=torch.float64
        )

        self.assertTrue(torch.equal(self.dataset.encodings[tid][0], expected_latlon),
                        f"Got {self.dataset.encodings[tid][0]}, expected {expected_latlon}.")
        self.assertTrue(torch.equal(self.dataset.encodings[tid][1], expected_hour),
                        f"Got {self.dataset.encodings[tid][1]}, expected {expected_hour}.")
        self.assertTrue(torch.equal(self.dataset.encodings[tid][2], expected_day),
                        f"Got {self.dataset.encodings[tid][2]}, expected {expected_day}.")

    @patch("pandas.read_csv")
    def test_original_dataframe_content_is_preserved(self, mock_read_csv):
        # Mocking a sample dataframe with a trajectory of length 5
        mock_df = pd.DataFrame({
            'tid': ['t1', 't1', 't1', 't1', 't1'],
            'uid': [1, 1, 1, 1, 1],
            LAT: [40.0, 40.1, 40.2, 40.3, 40.4],
            LON: [116.3, 116.4, 116.5, 116.6, 116.7],
            'altitude': [10.0, 11.0, 12.0, 13.0, 14.0],
            'timestamp': [
                pd.Timestamp('2022-01-01 12:00:00'),
                pd.Timestamp('2022-01-01 13:00:00'),
                pd.Timestamp('2022-01-01 14:00:00'),
                pd.Timestamp('2022-01-01 15:00:00'),
                pd.Timestamp('2022-01-01 16:00:00')
            ]
        })
        original_df_copy = mock_df.copy(deep=True)  # Create a deep copy
        mock_read_csv.return_value = mock_df

        tid = "t1"
        self.dataset.normalize = True  # Using normalization
        self.dataset.keep_original = True  # Storing the original dataframe

        self.dataset._encode(tid)

        # Assert that the content of the original dataframe matches the copy's content
        for column in original_df_copy.columns:
            pd.testing.assert_series_equal(self.dataset._originals[tid][column], original_df_copy[column])
