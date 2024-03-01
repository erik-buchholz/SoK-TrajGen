#!/usr/bin/env python3
"""Base module for Trajectory Generators. Contains joint functionality of LSTM-TrajGAN and STG."""
import json
import logging
import warnings
from abc import ABC
from pathlib import Path
from typing import Optional

import torch
from opacus.optimizers import DPOptimizer
from torch import nn
from torch.utils.data import DataLoader

from stg.models.utils import prepare_param_path
from stg.utils.helpers import compute_delta

log = logging.getLogger()


class TrajGAN(nn.Module, ABC):

    @property
    def device(self):
        """Assumes all parameters are on the same device."""
        return next(self.parameters()).device

    def get_delta(self) -> float:
        if self.dp:
            return self.delta
        else:
            raise RuntimeError("Model is trained without differential privacy.")

    def get_epsilon(self) -> float:
        if self.dp:
            return self.privacy_engine.get_epsilon(self.get_delta())
        else:
            raise RuntimeError("Model is trained without differential privacy.")

    def prepare_param_paths(self, name: str, param_path: Optional[str] = None):
        self.param_path = prepare_param_path(name=name, param_path=param_path)
        self.com_weight_path = self.param_path.format(EPOCH="{epoch:04d}", MODEL="COM")
        self.gen_weight_path = self.param_path.format(EPOCH="{epoch:04d}", MODEL="GEN")
        self.dis_weight_path = self.param_path.format(EPOCH="{epoch:04d}", MODEL="DIS")


    def save_parameters(self, epoch: int):
        # Create directory if it does not exist
        Path(self.com_weight_path).parent.mkdir(parents=True, exist_ok=True)
        if not self.dp:
            torch.save(self.gen.state_dict(), self.gen_weight_path.format(epoch=epoch))
            torch.save(self.dis.state_dict(), self.dis_weight_path.format(epoch=epoch))
            torch.save(self.state_dict(), self.com_weight_path.format(epoch=epoch))
            log.info(f"Saved parameters to {self.com_weight_path}")
        else:
            if not self.dp_initialised:
                raise RuntimeError("Checkpointing only possible after dp_init() has been called.")
            # Use privacy engine's checkpointing functionality
            # 1. Save discriminator as usual
            torch.save(self.dis.state_dict(), self.dis_weight_path.format(epoch=epoch))
            # 2. Save generator via privacy engine
            self.privacy_engine.save_checkpoint(
                path=self.gen_weight_path.format(epoch=epoch),
                module=self.gen,
                optimizer=self.opt_g,
            )
            # 3. Save DP parameters
            self.opt_g: DPOptimizer
            dp_params = {
                'delta': self.delta,
                'max_grad_norm': self.opt_g.max_grad_norm,
                'noise_multiplier': self.opt_g.noise_multiplier,
                'epochs': self.epochs,
                'target_epsilon': self.target_epsilon
            }
            with open(self.param_path + 'dp_params.json', 'w') as f:
                json.dump(dp_params, f)
            log.info(f"Saved parameters to {self.com_weight_path}.")

    def load_parameters(self, epoch: int, dataloader: DataLoader = None) -> DataLoader or None:
        if not self.dp:
            self.load_state_dict(torch.load(self.com_weight_path.format(epoch=epoch), map_location=self.device))
            log.info(f"Loaded parameters from {self.com_weight_path.format(epoch=epoch)}")
        else:
            if dataloader is None:
                raise RuntimeError("DataLoader has to be provided if loading DP model")
            # Load discriminator
            self.dis.load_state_dict(torch.load(self.dis_weight_path.format(epoch=epoch)))
            # Load dp parameters
            with open(self.param_path + 'dp_params.json', 'r') as f:
                dp_params = json.load(f)
            self.delta = dp_params['delta']
            self.epochs = dp_params['epochs']
            self.target_epsilon = dp_params['target_epsilon']
            # Load privacy engine & generator
            self.gen, self.opt_g, dataloader = self.privacy_engine.make_private(
                module=self.gen,
                optimizer=self.opt_g,
                data_loader=dataloader,
                noise_multiplier=dp_params['noise_multiplier'],
                max_grad_norm=dp_params['max_grad_norm']
            )
            self.privacy_engine.load_checkpoint(
                path=self.gen_weight_path.format(epoch=epoch),
                module=self.gen,
                optimizer=self.opt_g
            )
            log.info(f"Loaded parameters from {self.param_path}")
            return dataloader

    def init_dp(self, dataloader: DataLoader, epochs: int,
                max_grad_norm: float,
                noise_multiplier: float = None,
                target_epsilon: float = None,
                delta: float = None,
                ):
        log.info("Initializing privacy engine, might take some time.")
        # The privacy engine raises a few warning we cannot do anything about
        warnings.simplefilter("ignore")
        self.delta = compute_delta(len(dataloader.dataset)) if delta is None else delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs

        if target_epsilon is not None:
            self.target_epsilon = target_epsilon
            log.info(f"Targeting (ε = {self.target_epsilon}, δ = {self.delta})")
        elif noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
            self.target_epsilon = 0  # Important for saving checkpoints
            log.info(f"Utilizing (σ = {self.noise_multiplier}, δ = {self.delta})")
        else:
            raise ValueError("Either target_epsilon or noise_multiplier have to be provided!")

        compatible = self.privacy_engine.is_compatible(
            module=self.gen,
            optimizer=self.opt_g,
            data_loader=dataloader
        )
        if compatible:
            log.info("Model compatible with privacy settings!")
        else:
            raise RuntimeError("Model, Optimizer or dataset not compatible with DP-SGD.")

        if target_epsilon is not None:
            self.gen, self.opt_g, dataloader = self.privacy_engine.make_private_with_epsilon(
                module=self.gen,
                optimizer=self.opt_g,
                data_loader=dataloader,
                epochs=self.epochs,
                target_epsilon=self.target_epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True,
                # grad_sample_mode="ew"  # functorch
            )
            warnings.simplefilter("default")
        else:
            self.gen, self.opt_g, dataloader = self.privacy_engine.make_private(
                module=self.gen,
                optimizer=self.opt_g,
                data_loader=dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True
            )
        self.dp_initialised = True
        log.info(f"Using σ={self.opt_g.noise_multiplier} and C={self.max_grad_norm}")
        return dataloader
