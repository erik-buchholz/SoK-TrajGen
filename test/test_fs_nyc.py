#!/usr/bin/env python3
"""Unittest for Foursquare Dataset."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from stg.datasets.fs_nyc import FSNYCDataset
from stg.utils import data

# Constants used in the test
TID = "tid"
UID = "label"
LAT = "lat"
LON = "lon"
HOUR = "hour"
DAY = "day"
CAT = "cat"
SPATIAL_COLUMNS = [LON, LAT]
SPATIAL_FEATURE = "spatial_feature"
MAX_LEN = 100
PATH_ALL = "all.csv"
PATH_TRAIN = "train.csv"
PATH_TEST = "test.csv"

# Dummy data
DUMMY_DF = pd.DataFrame({
    TID: ['t1', 't1', 't1', 't1', 't1'],
    UID: [1, 1, 1, 1, 1],
    LAT: [40.0, 40.1, 40.2, 40.3, 40.4],
    LON: [116.3, 116.4, 116.5, 116.6, 116.7],
    HOUR: [12, 13, 14, 15, 16],
    DAY: [1, 2, 3, 4, 5],
    CAT: [1, 1, 2, 2, 3],
})


class Test(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.dataset_latlon = FSNYCDataset(keep_original=True, latlon_only=True)
        cls.dataset = FSNYCDataset(keep_original=True, latlon_only=False)

    def test_normalization_latlon(self):
        # Test that the normalization is reversible
        for _ in range(10):
            i = np.random.randint(0, len(self.dataset))
            tid = self.dataset_latlon.tids[i]
            normalized = self.dataset_latlon[i]
            original = self.dataset_latlon.originals[tid]

            denormalized = data.denormalize_points(
                normalized.numpy(),
                self.dataset_latlon.reference_point,
                self.dataset_latlon.scale_factor,
            )

            np.testing.assert_allclose(original[self.dataset_latlon.columns], denormalized, atol=1e-5)

    def test_normalization(self):
        # Test that the normalization is reversible
        for _ in range(10):
            i = np.random.randint(0, len(self.dataset))
            tid = self.dataset.tids[i]
            normalized = self.dataset[i]
            original = self.dataset.originals[tid]

            denormalized = data.denormalize_points(
                normalized[0].numpy(),
                self.dataset.reference_point,
                self.dataset.scale_factor,
            )

            np.testing.assert_allclose(original[self.dataset.columns], denormalized, atol=1e-5)

    @patch("pandas.read_csv", return_value=DUMMY_DF)
    def test_initialization(self, mock_read_csv):
        dataset = FSNYCDataset()
        self.assertTrue(hasattr(dataset, "originals"))
        self.assertTrue(hasattr(dataset, "tids"))
        self.assertTrue(hasattr(dataset, "encodings"))

    @patch("pandas.read_csv", return_value=DUMMY_DF)
    def test_original_dataframe_content_is_preserved(self, mock_read_csv):
        original = DUMMY_DF.copy(deep=True)
        dataset = FSNYCDataset(normalize=True, keep_original=True)
        for column in DUMMY_DF.columns:
            pd.testing.assert_series_equal(dataset.originals['t1'][column], original[column])

    @patch("pandas.read_csv", return_value=DUMMY_DF)
    def test_getitem(self, mock_read_csv):
        dataset = FSNYCDataset()
        item = dataset[0]
        self.assertIsInstance(item, torch.Tensor)
        self.assertEqual(item.shape, torch.Size([5, 2]))  # 5 rows for 5 time steps and 2 for lon and lat

        dataset = FSNYCDataset(return_labels=True)
        item = dataset[0]
        self.assertIsInstance(item, list)
        self.assertEqual(len(item), 2)
        self.assertIsInstance(item[0], torch.Tensor)
        self.assertEqual(item[0].shape, torch.Size([5, 2]))
        self.assertEqual(item[1], 't1')
