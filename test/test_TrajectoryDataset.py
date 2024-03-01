#!/usr/bin/env python3
""" """
import unittest
from unittest import TestCase

import pandas as pd
import torch

from stg.datasets.base_dataset import SubTrajectoryDataset, random_split
from stg.datasets.padding import pre_pad


class Test(TestCase):
    def test_pre_pad(self):
        lst = [
            torch.Tensor([1, 2, 3, 4, 5]),
            torch.Tensor([10, 20]),
        ]
        res = torch.Tensor([
            [1, 2, 3, 4, 5],
            [0, 0, 0, 10, 20],
        ])
        torch.testing.assert_close(
            res,
            pre_pad(lst, batch_first=True)
        )


class TestSubTrajectoryDataset(unittest.TestCase):

    def setUp(self):
        self.mock_dataset = unittest.mock.MagicMock()
        self.mock_dataset.tids = ['A', 'B', 'C', 'D', 'E']
        self.mock_dataset.uids = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
        self.mock_dataset.encodings = {'A': torch.tensor([1]), 'B': torch.tensor([2]), 'C': torch.tensor([3])}
        self.mock_dataset.originals = {'A': pd.DataFrame(), 'B': pd.DataFrame(), 'C': pd.DataFrame()}
        self.mock_dataset.mode = 'train'

    def test_initialization(self):
        subset = SubTrajectoryDataset(self.mock_dataset, [1, 2])
        self.assertEqual(len(subset), 2)
        self.assertIn('B', subset.tids)
        self.assertIn('C', subset.tids)

    def test_len(self):
        subset = SubTrajectoryDataset(self.mock_dataset, [0, 2])
        self.assertEqual(len(subset), 2)

    def test_getitem(self):
        subset = SubTrajectoryDataset(self.mock_dataset, [0, 2])
        item = subset[1]
        self.assertEqual(item, self.mock_dataset[2])

    def test_getattr(self):
        subset = SubTrajectoryDataset(self.mock_dataset, [0, 1])
        self.assertEqual(subset.mode, self.mock_dataset.mode)

    def test_proxy_behavior(self):
        subset = SubTrajectoryDataset(self.mock_dataset, [0, 1])
        self.assertEqual(subset.uids['A'], 1)
        with self.assertRaises(ValueError):
            subset.uids['E']


class TestRandomSplit(unittest.TestCase):

    def setUp(self):
        self.mock_dataset = unittest.mock.MagicMock()
        self.mock_dataset.tids = ['A', 'B', 'C', 'D', 'E']
        self.mock_dataset.__len__.return_value = 5

    def test_split_int_sizes(self):
        subsets = random_split(self.mock_dataset, [2, 3])
        self.assertEqual(len(subsets), 2)
        self.assertEqual(len(subsets[0]), 2)
        self.assertEqual(len(subsets[1]), 3)

    def test_split_float_sizes(self):
        subsets = random_split(self.mock_dataset, [0.4, 0.6])
        self.assertEqual(len(subsets), 2)
        self.assertEqual(len(subsets[0]), 2)
        self.assertEqual(len(subsets[1]), 3)

    def test_invalid_sizes(self):
        with self.assertRaises(AssertionError):
            random_split(self.mock_dataset, [3, 4])
        with self.assertRaises(AssertionError):
            random_split(self.mock_dataset, [0.5, 0.6])
        with self.assertRaises(ValueError):
            random_split(self.mock_dataset, [2, 0.5])

    def test_correct_splits(self):
        subsets = random_split(self.mock_dataset, [3, 2])
        total_length = sum([len(subset) for subset in subsets])
        self.assertEqual(total_length, 5)

        unique_tids = set()
        for subset in subsets:
            for tid in subset.tids:
                self.assertNotIn(tid, unique_tids, "Trajectory IDs should not overlap between subsets.")
                unique_tids.add(tid)
