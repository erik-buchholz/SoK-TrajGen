#!/usr/bin/env python3
"""Base RNN Model to create common interface."""
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from stg.datasets import Datasets
from stg.models import utils
from stg.models.utils import Optimizer

MODEL_STR = "GEN"


class BaseRNN(ABC, nn.Module):
    """Base RNN Model to create common interface."""

    def __init__(
            self,
            name: str,
            input_dim: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            rnn_type: utils.RNN_TYPES,
            bidirectional: bool,
            output_func: nn.Module,
            embedding_dim: int = None,
            output_dim: int = None,
    ):
        super().__init__()

        if name is None:
            raise ValueError("Model name must be specified.")

        # Store parameters
        self.input_dim = input_dim
        self.noise_dim = input_dim  # Only used by models receiving noise input
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.output_func = output_func
        if embedding_dim is None:
            self.embedding_dim = input_dim
        else:
            self.embedding_dim = embedding_dim
        if output_dim is None:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim
        self.name = name
        self.param_path = utils.prepare_param_path(name=name)

    # .device Property returns device of parameters
    @property
    def device(self) -> torch.device:
        """Makes assumption that all parameters are on the same device."""
        return next(self.parameters()).device

    @abstractmethod
    def training_loop(self,
                      dataloader: DataLoader,
                      epochs: int,
                      device: str,
                      dataset_name: Datasets,
                      lr: float = 0.001,
                      beta1: float = 0.9,
                      beta2: float = 0.999,
                      clip_value: float = 0.0,
                      plot_freq: int = 200,
                      save_freq: int = 100,
                      param_path: str = None,
                      notebook: bool = False,
                      tensorboard: bool = True,
                      opt: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
                      name: Optional[str] = None,
                      suffix: str = '',
                      **kwargs
                      ) -> None:
        pass

    def save_parameters(self, epoch: int):
        # Save model
        utils.save_model(self, epoch=epoch, param_path=self.param_path, model_str=MODEL_STR)

    def load_parameters(self, epoch: int, param_path: str = None):
        param_path = self.param_path if param_path is None else param_path
        # Load combined model
        utils.load_model(self, epoch=epoch, param_path=param_path, model_str=MODEL_STR, device=self.device)
