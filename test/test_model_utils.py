#!/usr/bin/env python3
"""Test Cases for stg.models.utils"""
import unittest

import numpy as np
import torch
from torch import nn, Tensor

from stg.models import utils
from stg.models.utils import merge_bilstm_output, compute_mask_from_lengths


class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_mask(self):
        real_traj = torch.tensor([
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ])
        mask = utils.compute_mask(real_traj)
        self.assertEqual(mask.shape, (3, 3))
        self.assertTrue(torch.all(mask == torch.tensor([[False, False, False],
                                                        [False, False, False],
                                                        [False, False, False]])))
        real_traj = torch.tensor([
            [
                [5, 4],
                [5, 9],
                [6, 3]
            ],
            [
                [0, 0],
                [3, 6],
                [0, 0]
            ],
            [
                [0, 0],
                [0, 0],
                [5, 5]
            ]])
        mask = utils.compute_mask(real_traj)
        self.assertEqual(mask.shape, (3, 3))
        self.assertTrue(torch.all(mask == torch.tensor([[True, True, True],
                                                        [False, True, False],
                                                        [False, False, True]])))
        self.assertTrue(
            torch.all(
                real_traj * mask.unsqueeze(-1) == real_traj
            ),
            f"Mask: {mask.shape}, Real Traj: {real_traj.shape}"
        )

        # Test multiplication
        x = torch.randn((3, 3, 2))
        masked_x = x * mask.unsqueeze(-1)
        expected_masked_x = x
        expected_masked_x[1, 0] = torch.tensor([0.0, 0.0])
        expected_masked_x[1, 2] = torch.tensor([0.0, 0.0])
        expected_masked_x[2, 0] = torch.tensor([0.0, 0.0])
        expected_masked_x[2, 1] = torch.tensor([0.0, 0.0])
        self.assertTrue(torch.all(masked_x == expected_masked_x))


class DiscriminatorMock(nn.Module):
    def __init__(self):
        super(DiscriminatorMock, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.linear.weight.data = torch.tensor([[1.0, 1.0]])
        self.linear.bias.data = torch.tensor([0.0])

    def forward(self, x, lengths=None):
        # Mock behavior of discriminator.
        if isinstance(x, Tensor):
            return self.linear(x)
        else:
            # Assume x is a list of tensors
            out = torch.sum(torch.cat(x, dim=2), dim=-1, keepdim=True)
            assert out.dim() == 3 and out.shape[-1] == 1
            return out


class TestGradientPenalty(unittest.TestCase):

    def setUp(self):
        # Set up a simple mock discriminator
        self.discriminator = DiscriminatorMock()

        # Create fake and real inputs for testing
        self.batch_size = 4
        self.sequence_length = 10
        self.feature_dim = 2
        self.real_samples = torch.randn((self.batch_size, self.sequence_length, self.feature_dim), requires_grad=True)
        self.synthetic_samples = torch.randn((self.batch_size, self.sequence_length, self.feature_dim),
                                             requires_grad=True)

        # Create samples with concrete number for deterministic testing
        # Use batch size 3, length 5, feature dim 2
        self.real_samples_det = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0], [0.0, 0.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0], [0.0, 0.0]]])
        self.synthetic_samples_det = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0], [0.0, 0.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0], [0.0, 0.0]]])

        # Create optional lengths list
        self.lengths = [self.sequence_length] * self.batch_size

        # Create multi-feature samples
        feature_dims = [2, 3, 4]
        self.real_samples_multi = [torch.randn((self.batch_size, self.sequence_length, d)) for d in feature_dims]
        self.synthetic_samples_multi = [torch.randn((self.batch_size, self.sequence_length, d)) for d in feature_dims]

        # Create multi-feature samples with concrete number for deterministic testing
        self.real_samples_multi_det = [
            torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0], [0.0, 0.0]],
                          [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
                          [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0], [0.0, 0.0]]]),
            torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                          [[7.0, 8.0, 9.0], [9.0, 10.0, 11.0], [11.0, 12.0, 13.0], [13.0, 14.0, 15.0],
                           [15.0, 16.0, 17.0]],
                          [[17.0, 18.0, 19.0], [19.0, 20.0, 21.0], [21.0, 22.0, 23.0], [23.0, 24.0, 25.0],
                           [0.0, 0.0, 0.0]]]),
        ]
        self.synthetic_samples_multi_det = [
            torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0], [0.0, 0.0]],
                          [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
                          [[54.0, 18.0], [54.0, 20.0], [54.0, 22.0], [54.0, 24.0], [0.0, 0.0]]]),
            torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 54.0, 7.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                          [[7.0, 8.0, 9.0], [9.54, 10.0, 11.0], [11.0, 12.0, 13.0], [54.0, 14.0, 15.0],
                           [15.0, 16.0, 17.0]],
                          [[17.0, 18.0, 19.0], [19.0, 20.0, 21.0], [21.0, 54.0, 23.0], [23.0, 24.0, 25.0],
                           [0.0, 0.0, 0.0]]]),
        ]

    def test_gradient_penalty(self):
        # Test with standard gradient penalty
        gp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples,
            self.synthetic_samples,
            lengths=None,
            lp=False
        )
        self.assertIsInstance(gp, torch.Tensor)

    def test_gradient_penalty_multi(self):
        # Test with standard gradient penalty
        gp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples_multi,
            self.synthetic_samples_multi,
            lengths=None,
            lp=False
        )
        self.assertIsInstance(gp, torch.Tensor)

    def test_lipschitz_penalty(self):
        # Test with Lipschitz penalty
        lp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples,
            self.synthetic_samples,
            lengths=None,
            lp=True
        )
        self.assertIsInstance(lp, torch.Tensor)

    def test_lipschitz_penalty_multi(self):
        # Test with Lipschitz penalty
        lp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples_multi,
            self.synthetic_samples_multi,
            lengths=None,
            lp=True
        )
        self.assertIsInstance(lp, torch.Tensor)

    def test_length_handling(self):
        # Test handling of the lengths parameter
        gp_with_length = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples,
            self.synthetic_samples,
            lengths=self.lengths,
            lp=False
        )
        self.assertIsInstance(gp_with_length, torch.Tensor)

    def test_length_handling_multi(self):
        # Test handling of the lengths parameter
        gp_with_length = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples_multi,
            self.synthetic_samples_multi,
            lengths=self.lengths,
            lp=False
        )
        self.assertIsInstance(gp_with_length, torch.Tensor)

    def test_penalty_values(self):
        # Check if the penalties are reasonable (not trivially zero, positive, etc.)
        gp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples,
            self.synthetic_samples,
            lengths=None,
            lp=False
        )
        self.assertGreater(gp.item(), 0, "Gradient penalty should be greater than 0")

        lp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples,
            self.synthetic_samples,
            lengths=None,
            lp=True
        )
        self.assertGreater(lp.item(), 0, "Lipschitz penalty should be greater than 0")

    def test_penalty_values_multi(self):
        # Check if the penalties are reasonable (not trivially zero, positive, etc.)
        gp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples_multi,
            self.synthetic_samples_multi,
            lengths=None,
            lp=False
        )
        self.assertGreater(gp.item(), 0, "Gradient penalty should be greater than 0")

        lp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples_multi,
            self.synthetic_samples_multi,
            lengths=None,
            lp=True
        )
        self.assertGreater(lp.item(), 0, "Lipschitz penalty should be greater than 0")

    def test_deterministic_values(self):
        # Check if the penalties are reasonable (not trivially zero, positive, etc.)
        gp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples_det,
            self.synthetic_samples_det,
            lengths=None,
            lp=False
        )
        self.assertAlmostEqual(gp.item(), 4.675445, places=5,
                               msg="Gradient penalty not as expected")

        lp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples_det,
            self.synthetic_samples_det,
            lengths=None,
            lp=True
        )
        self.assertAlmostEqual(lp.item(), 4.675445, places=5,
                               msg="Lipschitz penalty not as expected")

    def test_deterministic_values_multi(self):
        # Check if the penalties are reasonable (not trivially zero, positive, etc.)
        gp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples_multi_det,
            self.synthetic_samples_multi_det,
            lengths=None,
            lp=False
        )
        self.assertAlmostEqual(gp.item(), 4.675445, places=5,
                               msg="Gradient penalty not as expected")

        lp = utils.compute_gradient_penalty(
            self.discriminator,
            self.real_samples_multi_det,
            self.synthetic_samples_multi_det,
            lengths=None,
            lp=True
        )
        self.assertAlmostEqual(lp.item(), 4.675445, places=5,
                               msg="Lipschitz penalty not as expected")


class TestMergeBiLSTMOutput(unittest.TestCase):

    def test_sum_mode(self):
        """Test merging with sum mode."""
        x = torch.randn(3, 4, 6)  # Example tensor
        merged = merge_bilstm_output(x, 'sum')
        expected = torch.sum(x.view(3, 4, 2, 3), dim=2)
        torch.testing.assert_close(merged, expected)

    def test_average_mode(self):
        """Test merging with average mode."""
        x = torch.randn(3, 4, 6)  # Example tensor
        merged = merge_bilstm_output(x, 'average')
        expected = torch.mean(x.view(3, 4, 2, 3), dim=2)
        torch.testing.assert_close(merged, expected)

    def test_concat_mode(self):
        """Test merging with concat mode."""
        x = torch.randn(3, 4, 6)  # Example tensor
        merged = merge_bilstm_output(x, 'concat')
        expected = x  # In concat mode, x should remain unchanged
        torch.testing.assert_close(merged, expected)

    def test_mul_mode(self):
        """Test merging with mul mode."""
        x = torch.randn(3, 4, 6)  # Example tensor
        merged = merge_bilstm_output(x, 'mul')
        expected = torch.prod(x.view(3, 4, 2, 3), dim=2)
        torch.testing.assert_close(merged, expected)

    def test_invalid_mode(self):
        """Test merging with an invalid mode."""
        x = torch.randn(3, 4, 6)  # Example tensor
        with self.assertRaises(ValueError):
            merge_bilstm_output(x, 'invalid')


class TestComputeMaskFromLengths(unittest.TestCase):

    def test_basic_functionality(self):
        """Test basic functionality with valid inputs."""
        x = torch.randn(4, 5, 3)  # Batch size 4, Sequence length 5, Feature channels 3
        lengths = [5, 3, 4, 1]
        expected_mask = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0]
        ])
        mask = compute_mask_from_lengths(x, lengths)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_empty_tensor(self):
        """Test functionality with an empty tensor."""
        x = torch.randn(0, 5, 3)  # Empty batch
        lengths = []
        with self.assertRaises(ValueError):
            compute_mask_from_lengths(x, lengths)

    def test_full_length(self):
        """Test functionality when all lengths are equal to the sequence length."""
        x = torch.randn(3, 4, 2)
        lengths = [4, 4, 4]
        expected_mask = torch.ones(3, 4, dtype=torch.bool)
        mask = compute_mask_from_lengths(x, lengths)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_zero_length(self):
        """Test functionality when some lengths are zero."""
        x = torch.randn(3, 4, 2)
        lengths = [0, 4, 2]
        expected_mask = torch.tensor([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 0]
        ])
        mask = compute_mask_from_lengths(x, lengths)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_invalid_lengths(self):
        """Test functionality with invalid lengths (greater than sequence length)."""
        x = torch.randn(2, 3, 4)
        lengths = [4, 5]  # 5 is greater than the sequence length of 3
        with self.assertRaises(ValueError):
            compute_mask_from_lengths(x, lengths)
