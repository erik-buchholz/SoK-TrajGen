#!/usr/bin/env python3
"""BaseClass for GANs"""
from abc import ABC, abstractmethod
from typing import Optional, Union, List

import torch
from torch import nn

from stg.models import utils


class BaseGAN(ABC, nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name is None:
            self.param_path = None
        else:
            self.param_path = utils.prepare_param_path(name=name)

    @abstractmethod
    def forward(self, x, lengths: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def training_loop(self, *args, **kwargs) -> None:
        pass

    @property
    def device(self):
        """Assumes that all parameters are on the same device"""
        return next(self.parameters()).device

    def save_parameters(self, epoch: int, save_individual: bool = False):
        # Save individual models
        if save_individual:
            utils.save_models(gen=self.gen, dis=self.dis, epoch=epoch, param_path=self.param_path)
        # Save combined model
        utils.save_model(self, epoch=epoch, param_path=self.param_path, model_str="COM")

    def load_parameters(self, epoch: int, param_path: str = None):
        param_path = self.param_path if param_path is None else param_path
        if param_path is None:
            raise ValueError("No param_path specified.")
        # Load combined model
        utils.load_model(self, epoch=epoch, param_path=param_path, model_str="COM", device=self.device)
