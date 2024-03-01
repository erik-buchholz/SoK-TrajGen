#!/usr/bin/env python3
"""
Creates MNIST sequential as PyTorch dataset with variable format.
This dataset is used by RGAN as default dataset.
"""
import torch
from torchvision import datasets, transforms

from stg import config

# Dataset name
MNISTSequential = 'mnist_sequential'


class Reshape:

    def __init__(self, size):
        self.shape = size

    def __call__(self, tensor):
        return torch.reshape(tensor, shape=self.shape)


def mnist_sequential(dim=28):
    if not isinstance(dim, int):
        raise ValueError(f"dim must be int, got {type(dim)}")
    mnist = datasets.MNIST(
        root=config.BASE_DIR + '/data/mnist', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5),
                                      Reshape(size=(28 * 28 // dim, dim))]))
    # Make this a TrajectoryDataset
    mnist.name = MNISTSequential
    mnist.features = ['mnist']
    return mnist
