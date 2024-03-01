#!/usr/bin/env python3
""" """
from unittest import TestCase

import numpy as np
import pandas as pd

from stg.utils import data


class Test(TestCase):

    def test_normalization_numpy_array(self):
        input_array = np.array([1, 2, 3, 4, 5])
        expected_normalized = np.array([-1, -0.5, 0.0, 0.5, 1])
        normalized, _, _ = data.normalize_sequence(input_array)
        np.testing.assert_allclose(normalized, expected_normalized)

    def test_normalization_pandas_series(self):
        input_series = pd.Series([10, 20, 30, 40, 50])
        expected_normalized = pd.Series([-1, -0.5, 0.0, 0.5, 1])
        normalized, _, _ = data.normalize_sequence(input_series)
        pd.testing.assert_series_equal(normalized, expected_normalized)

    def test_inplace_normalization(self):
        input_array = np.array([100., 200, 300, 400, 500])
        normalized_inplace, _, _ = data.normalize_sequence(input_array, inplace=True)
        np.testing.assert_allclose(input_array, normalized_inplace)

    def test_reference_and_scale_factor(self):
        input_array = np.array([2, 4, 6, 8, 10])
        _, ref, scale_factor = data.normalize_sequence(input_array)
        self.assertAlmostEqual(ref, 6.0)
        self.assertAlmostEqual(scale_factor, 4.0)

    def test_denormalization_numpy_array(self):
        input_array = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
        ref = 5.0
        scale_factor = 2.0
        expected_denormalized = np.array([4.0, 4.5, 5.0, 5.5, 6.0])
        denormalized = data.denormalize_sequence(input_array, ref, scale_factor)
        np.testing.assert_allclose(denormalized, expected_denormalized)

    def test_denormalization_pandas_series(self):
        input_series = pd.Series([-0.5, -0.25, 0.0, 0.25, 0.5])
        ref = 100.0
        scale_factor = 50.0
        expected_denormalized = pd.Series([75, 87.5, 100.0, 112.5, 125.0])
        denormalized = data.denormalize_sequence(input_series, ref, scale_factor)
        pd.testing.assert_series_equal(denormalized, expected_denormalized)

    def test_inplace_denormalization(self):
        input_array = np.array([1.0, 2.0, 3.0])
        ref = -1.0
        scale_factor = 0.5
        denormalized_inplace = data.denormalize_sequence(input_array, ref, scale_factor, inplace=True)
        np.testing.assert_allclose(input_array, denormalized_inplace)

    def test_normalize_denormalize_loop(self):
        init = np.random.randn(100) * 100
        normalized, ref, scale_factor = data.normalize_sequence(init)
        denormalized = data.denormalize_sequence(normalized, ref, scale_factor)
        np.testing.assert_allclose(init, denormalized)

    def test_normalize_points_pandas_dataframe(self):
        input_data = {'x': [2, 4, 6, 8, 10], 'y': [100, 200, 300, 400, 500]}
        input_df = pd.DataFrame(input_data)
        expected_normalized_data = {'x': [-1.0, -0.5, 0.0, 0.5, 1.0], 'y': [-1.0, -0.5, 0.0, 0.5, 1.0]}
        expected_references = [6.0, 300.0]
        expected_scale_factors = [4.0, 200.0]

        normalized_df, references, scale_factors = data.normalize_points(input_df)
        pd.testing.assert_frame_equal(normalized_df, pd.DataFrame(expected_normalized_data))
        self.assertListEqual(references, expected_references)
        self.assertListEqual(scale_factors, expected_scale_factors)
        # Verify that the original dataframe is not modified
        pd.testing.assert_frame_equal(input_df, pd.DataFrame(input_data))

    def test_normalize_points_numpy_array(self):
        input_data = np.array([[2, 100], [4, 200], [6, 300], [8, 400], [10, 500]])
        expected_normalized_data = np.array([[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        expected_references = [6.0, 300.0]
        expected_scale_factors = [4.0, 200.0]

        normalized_data, references, scale_factors = data.normalize_points(input_data, columns=[0, 1])
        np.testing.assert_allclose(normalized_data, expected_normalized_data)
        self.assertListEqual(references, expected_references)
        self.assertListEqual(scale_factors, expected_scale_factors)
        # Verify that the original array is not modified
        np.testing.assert_allclose(input_data, np.array([[2, 100], [4, 200], [6, 300], [8, 400], [10, 500]]))

    def test_inplace_normalize_points_numpy(self):
        input_data = np.array([[1., 100], [2, 200], [3, 300]])
        normalized_data_inplace, _, _ = data.normalize_points(input_data, columns=[0, 1], inplace=True)
        np.testing.assert_allclose(input_data, normalized_data_inplace)

    def test_inplace_normalize_points_pandas(self):
        input_data = pd.DataFrame({'x': [1., 2, 3], 'y': [100, 200, 300]})
        normalized_data_inplace, _, _ = data.normalize_points(input_data, inplace=True)
        pd.testing.assert_frame_equal(input_data, normalized_data_inplace)

    def test_denormalize_points_pandas_dataframe(self):
        input_data = {'x': [-1.0, -0.5, 0.0, 0.5, 1.0], 'y': [-1.0, -0.5, 0.0, 0.5, 1.0]}
        input_df = pd.DataFrame(input_data)
        references = [6.0, 300.0]
        scale_factors = [4.0, 200.0]
        expected_denormalized_data = [[2., 100], [4, 200], [6, 300], [8, 400], [10, 500]]

        denormalized_df = data.denormalize_points(input_df[['x', 'y']].to_numpy(), references, scale_factors)
        np.testing.assert_allclose(denormalized_df, expected_denormalized_data)

    def test_denormalize_points_inplace(self):
        input_data = np.array([[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        ref = (6.0, 300.0)
        scale = (4.0, 200.0)
        expected_denormalized_data = np.array([[2.0, 100.0], [4.0, 200.0], [6.0, 300.0], [8.0, 400.0], [10.0, 500.0]])

        denormalized_data = data.denormalize_points(input_data, ref, scale, inplace=True)
        np.testing.assert_allclose(denormalized_data, expected_denormalized_data)

    def test_denormalize_points_not_inplace(self):
        input_data = np.array([[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        ref = (6.0, 300.0)
        scale = (4.0, 200.0)
        expected_denormalized_data = np.array([[2.0, 100.0], [4.0, 200.0], [6.0, 300.0], [8.0, 400.0], [10.0, 500.0]])

        denormalized_data = data.denormalize_points(input_data, ref, scale, inplace=False)
        np.testing.assert_allclose(denormalized_data, expected_denormalized_data)

    def test_denormalize_points_with_custom_columns(self):
        input_data = np.array([[0.0, 1.0, 10.0], [0.5, 0.5, 20.0], [1.0, 0.0, 30.0]])
        ref = (1.0, 10.0)
        scale = (0.5, 10.0)
        columns = [1, 2]
        expected_denormalized_data = np.array([[0.0, 1.5, 110], [0.5, 1.25, 210], [1.0, 1.0, 310]])

        denormalized_data = data.denormalize_points(input_data, ref, scale, columns=columns, inplace=False)
        np.testing.assert_allclose(denormalized_data, expected_denormalized_data)
