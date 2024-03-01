#!/usr/bin/env python3
"""Test for stg.models.layers"""
import unittest
import torch
import torch.nn as nn
from stg.models.layers import Norm, Activation, CustomLinear


class TestNorm(unittest.TestCase):
    def test_batchnorm1d_initialization(self):
        """Test initialization with BatchNorm1d."""
        norm = Norm(10, 'batch1d')
        self.assertIsInstance(norm.norm, nn.BatchNorm1d)

    def test_layernorm_initialization(self):
        """Test initialization with LayerNorm."""
        norm = Norm(10, 'layer')
        self.assertIsInstance(norm.norm, nn.LayerNorm)

    def test_identity_initialization(self):
        """Test initialization with Identity."""
        norm = Norm(10, 'none')
        self.assertIsInstance(norm.norm, nn.Identity)

    def test_dropout_initialization(self):
        """Test initialization with Dropout."""
        norm = Norm(10, 'dropout')
        self.assertIsInstance(norm.norm, nn.Dropout)

    def test_unknown_norm_type(self):
        """Test initialization with an unknown norm type."""
        with self.assertRaises(ValueError):
            Norm(10, 'unknown')

    def test_forward_with_batchnorm(self):
        """Test forward pass with BatchNorm1d."""
        norm = Norm(10, 'batch1d')
        x = torch.randn(5, 10, 20)  # Example shape (N, C, L)
        y = norm(x)
        self.assertEqual(x.shape, y.shape)

    def test_forward_with_layernorm(self):
        """Test forward pass with LayerNorm."""
        norm = Norm(10, 'layer', channels_last=False)
        x = torch.randn(5, 10, 20)  # Example shape (N, C, L)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_forward_with_layernorm_channels_last(self):
        """Test forward pass with LayerNorm and channels_last=True."""
        norm = Norm(10, 'layer', channels_last=True)
        x = torch.randn(5, 20, 10)  # Example shape (N, L, C)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)


class TestActivation(unittest.TestCase):

    def test_relu_initialization(self):
        """Test initialization with ReLU activation."""
        activation = Activation('relu')
        self.assertIsInstance(activation.activation, nn.ReLU)

    def test_leaky_relu_initialization(self):
        """Test initialization with LeakyReLU activation."""
        activation = Activation('leaky_relu')
        self.assertIsInstance(activation.activation, nn.LeakyReLU)

    def test_unknown_activation(self):
        """Test initialization with an unknown activation."""
        with self.assertRaises(ValueError):
            Activation('unknown')

    def test_relu_forward(self):
        """Test forward pass with ReLU activation."""
        activation = Activation('relu')
        x = torch.tensor([-1.0, 0.0, 1.0])
        y = activation(x)
        expected = torch.tensor([0.0, 0.0, 1.0])
        torch.testing.assert_close(y, expected)

    def test_leaky_relu_forward(self):
        """Test forward pass with LeakyReLU activation."""
        activation = Activation('leaky_relu')
        x = torch.tensor([-1.0, 0.0, 1.0])
        y = activation(x)
        expected = torch.tensor([-0.2, 0.0, 1.0])  # Using 0.2 as the negative slope
        torch.testing.assert_close(y, expected)

    def test_string_representation(self):
        """Test the string representation of the activation module."""
        relu_activation = Activation('relu')
        self.assertEqual(str(relu_activation), 'ReLU()')

        leaky_relu_activation = Activation('leaky_relu')
        self.assertEqual(str(leaky_relu_activation), 'LeakyReLU(negative_slope=0.2)')


class TestCustomLinear(unittest.TestCase):

    def test_linear_initialization(self):
        """Test initialization with linear layer."""
        custom_linear = CustomLinear(10, 5, mode='linear')
        self.assertIsInstance(custom_linear.layer, nn.Linear)

    def test_conv1d_initialization(self):
        """Test initialization with conv1d layer."""
        custom_linear = CustomLinear(10, 5, mode='conv1d')
        self.assertIsInstance(custom_linear.layer, nn.Conv1d)

    def test_unknown_layer_type(self):
        """Test initialization with an unknown layer type."""
        with self.assertRaises(ValueError):
            CustomLinear(10, 5, mode='unknown')

    def test_forward_linear_fixed_weights(self):
        """Test forward pass with linear layer and fixed weights."""
        custom_linear = CustomLinear(2, 3, mode='linear')
        custom_linear.layer.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        custom_linear.layer.bias.data = torch.tensor([1.0, 2.0, 3.0])
        x = torch.tensor([[1.0, 2.0]])
        y = custom_linear(x)
        expected = torch.tensor([[6.0, 13.0, 20.0]])  # (1*1 + 2*2) + 1, (1*3 + 2*4) + 2, (1*5 + 2*6) + 3
        torch.testing.assert_close(y, expected)

    def test_forward_conv1d_fixed_weights(self):
        """Test forward pass with conv1d layer and fixed weights."""
        custom_linear = CustomLinear(2, 3, mode='conv1d')
        custom_linear.layer.weight.data = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
        custom_linear.layer.bias.data = torch.tensor([1.0, 2.0, 3.0])
        x = torch.tensor([[1.0, 2.0]])
        y = custom_linear(x)
        expected = torch.tensor([[6.0, 13.0, 20.0]])  # (1*1 + 2*2) + 1, (1*3 + 2*4) + 2, (1*5 + 2*6) + 3
        torch.testing.assert_close(y, expected)

    def test_equality_of_layers(self):
        """Test if the linear and conv1d layers are equivalent."""
        linear = CustomLinear(10, 5, mode='linear')
        conv1d = CustomLinear(10, 5, mode='conv1d')
        # Set weights and biases to the same values
        linear.layer.weight.data = conv1d.layer.weight.data.reshape(linear.layer.weight.shape)
        linear.layer.bias.data = conv1d.layer.bias.data.reshape(linear.layer.bias.shape)
        x = torch.randn(5, 10)
        y_linear = linear(x)
        y_conv1d = conv1d(x)
        torch.testing.assert_close(y_linear, y_conv1d)
