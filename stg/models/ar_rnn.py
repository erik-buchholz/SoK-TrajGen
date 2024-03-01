#!/usr/bin/env python3
"""
This module contains a very simple autoregressive LSTM model.
"""
import logging
from datetime import datetime
from typing import Optional, List

import numpy as np
import torch
from IPython import display
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_ntbk

from stg.datasets import Datasets, TrajectoryDataset
from stg.datasets.padding import ZeroPadding
from stg.models import utils
from stg.models.base_rnn import BaseRNN
from stg.models.utils import Optimizer, get_optimizer, prepare_param_path, NullWriter, split_data, \
    visualize_trajectory_samples, compute_mask, visualize_mnist_sequential

log = logging.getLogger(__name__)


class AR_RNN(BaseRNN):
    """
    AutoRegressive Recurrent Neural Network.
    Not only the hidden and cell state are passed to the next time step, but also the final
    post-processed output of the previous time step.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_size: int,
                 embedding_dim: int = None,
                 output_dim: int = None,
                 num_layers: int = 1,
                 rnn_type: utils.RnnType = utils.RnnType.LSTM,
                 bidirectional: bool = False,
                 output_func: nn.Module or None = nn.Tanh,
                 dropout: float = 0.0,
                 name: Optional[str] = "AR_RNN",
                 ):
        super().__init__(
            name=name,
            input_dim=input_dim,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            output_func=output_func,
        )

        if bidirectional:
            raise NotImplementedError("Bidirectional RNNs are not implemented yet.")

        if num_layers > 1:
            raise NotImplementedError("Multi-layer RNNs are not implemented yet.")

        if self.input_dim != self.output_dim:
            raise ValueError(f"Expected input_dim == output_dim, got {input_dim} != {output_dim}.")

        if dropout != 0:
            raise NotImplementedError("Dropout is not implemented yet.")

        self.start_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.Tanh()
        )
        self.rnn_cell = utils.get_rnn_cell(self.rnn_type, input_size=self.embedding_dim,
                                           hidden_size=self.hidden_size)

        self.output = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_dim)
        )
        self.last_layer = self.output_func() if output_func is not None else None

    def forward(self, noise: Tensor, length: int) -> Tensor:
        """
        Prediction Routine.
        :param noise: Noise (Tensor) w/ shape (batch_size, input_dim)
        :param length: Length of the output sequences
        :return: outputs.shape = (batch_size, time_steps = length, output_dim)
        """
        # Input shape: (batch_size, input_size)
        if not isinstance(noise, Tensor) or noise.dim() != 2:
            raise ValueError(f"Expected noise to be Tensor w/ dim=2, got {noise.shape}")

        batch_size = noise.size(0)
        outputs = []

        # First step receives noise as input
        ox = self.start_layer(noise)
        hx = torch.zeros(batch_size, self.hidden_size, dtype=noise.dtype, device=noise.device)
        cx = None
        if isinstance(self.rnn_cell, nn.LSTM) or isinstance(self.rnn_cell, nn.LSTMCell):
            # LSTM has two states while GRU has only one
            cx = torch.zeros_like(hx)

        for _ in range(length):
            if isinstance(self.rnn_cell, nn.LSTM) or isinstance(self.rnn_cell, nn.LSTMCell):
                hx, cx = self.rnn_cell(ox, (hx, cx))
            else:  # GRU or other cell types
                hx = self.rnn_cell(ox, hx)
            ox = self.output(hx)
            if self.last_layer is not None:
                ox = self.last_layer(ox)
            outputs.append(ox)

        # Batch size is second dimension now -> Switch
        outputs = torch.stack(outputs)
        outputs = torch.transpose(outputs, 0, 1)

        # Make sure output is of correct shape
        assert len(outputs) == batch_size, f"Expected batch size {batch_size}, got {len(outputs)}."
        assert outputs.size(1) == length, f"Expected length {length}, got {outputs.size(1)}."

        return outputs

    def training_step(self, noise: Tensor, length: int, real_samples: Tensor) -> Tensor:
        """
        Training Routine.
        :param noise: Noise (Tensor) w/ shape (batch_size, input_size)
        :param length: Length of the output sequences
        :param real_samples: Real samples (Tensor) used for teacher forcing
        :return: outputs.shape = (batch_size, time_steps = length, output_dim)
        """
        # Input shape: (batch_size, input_size)
        if not isinstance(noise, Tensor) or noise.dim() != 2:
            raise ValueError(f"Expected noise to be Tensor w/ dim=2, got {noise.shape}")

        batch_size = noise.size(0)
        outputs = []

        # First step receives noise as input
        ox = self.start_layer(noise)
        hx = torch.zeros(batch_size, self.hidden_size, dtype=noise.dtype, device=noise.device)
        cx = None
        if isinstance(self.rnn_cell, nn.LSTMCell) or isinstance(self.rnn_cell, nn.LSTM):
            # LSTM has two states while GRU has only one
            cx = torch.zeros_like(hx)

        for i in range(length):
            if isinstance(self.rnn_cell, nn.LSTMCell) or isinstance(self.rnn_cell, nn.LSTM):
                hx, cx = self.rnn_cell(ox, (hx, cx))
            else:  # GRU or other cell types
                hx = self.rnn_cell(ox, hx)
            ox = self.output(hx)
            if self.last_layer is not None:
                ox = self.last_layer(ox)
            outputs.append(ox)

            # Replace the output with the real value for the next step
            ox = real_samples[:, i]

        # Batch size is second dimension now -> Switch
        outputs = torch.stack(outputs)
        outputs = torch.transpose(outputs, 0, 1)

        # Make sure output is of correct shape
        assert len(outputs) == batch_size, f"Expected batch size {batch_size}, got {len(outputs)}."
        assert outputs.size(1) == length, f"Expected length {length}, got {outputs.size(1)}."

        return outputs

    def training_loop(
            self,
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
            suffix: str = "",
            **kwargs
    ) -> None:
        """
        Train an Autoregressive Recurrent Neural Network (AR_RNN).

        :param self: AR_RNN object
        :param dataloader: DataLoader.
        :param epochs: Number of epochs to train
        :param device: Device to use, e.g., 'cuda:0' or 'cpu'
        :param dataset_name: Name of the dataset
        :param lr: Learning Rate (default: 0.001)
        :param beta1: Beta1 for Adam optimizer (default: 0.9)
        :param beta2: Beta2 for Adam optimizer (default: 0.999)
        :param clip_value: Clipping value for gradients (default: 0 = DEACTIVATED)
        :param plot_freq: Plot generated samples after x batches. -1 to deactivate. (default: 200)
        :param save_freq: Save model after x epochs. -1 to deactivate. (default: 100)
        :param param_path: Directory to contain model parameters. Files will be param_path/EPOCH_MODEL.pth.
        :param notebook: Use the notebook version of TQDM and print to notebook instead of TensorBoard.
        :param tensorboard: Use TensorBoard for logging?
        :param opt: Optimizer to use. Default 'auto' uses recommended optimizer.
        :param name: Name of the model (for TensorBoard)
        :param suffix: Suffix for Model Name in Tensorboard
        :return:
        """
        # Input checks
        for key, value in kwargs.items():
            raise ValueError(f"Unused argument: {key}={value}")
        # Warn if clip value is set
        if clip_value != 0:
            log.warning(f"Gradient clipping is activated with value {clip_value}.")

        # Move to device
        self.to(device=device)

        # Determine Optimizer to use
        if not isinstance(opt, torch.optim.Optimizer):
            opt = get_optimizer(parameters=self.parameters(), choice=opt, lr=lr, beta_1=beta1, beta_2=beta2)

        # Get name for TensorBoard logging & parameter path
        if name is None:
            suffix = f"_{suffix}" if suffix != "" else ""
            name = f"AR_RNN_{self.rnn_type}_{dataset_name}_{lr}" + suffix
        self.name = name

        # Determine Parameter Path and create directory
        self.param_path = prepare_param_path(name=name, param_path=param_path)

        # Create Tensorboard connection
        writer = SummaryWriter('runs/' + datetime.now().strftime('%b-%d_%X_') + name) if tensorboard else NullWriter()

        # Determine progressbar function
        tqdm_func = tqdm_ntbk if notebook else tqdm

        # Loss function
        criterion = nn.MSELoss()

        # Create progressbar for epochs
        pbar_epochs = tqdm_func(range(1, epochs + 1), desc="Epochs")
        for epoch in pbar_epochs:
            # Create progressbar for batches
            pbar_batches = tqdm_func(enumerate(dataloader), leave=False, total=len(dataloader), desc="Batches")
            for i, data in pbar_batches:

                global_step = (epoch - 1) * len(dataloader) + i

                real_samples, lengths, labels = split_data(data)

                # Deal with datasets with constant lengths (lengths = None),\
                # i.e., all samples have the same maximum length
                if lengths is None:
                    lengths = [real_samples.shape[1]] * real_samples.shape[0]

                # Configure input
                real_samples = real_samples.to(device=device, dtype=torch.float32)

                # Zero the parameter gradients
                opt.zero_grad()

                # Generate noise
                noise = torch.randn(real_samples.size(0), self.noise_dim, device=device)

                # Forward pass
                outputs = self.training_step(noise=noise, length=real_samples.size(1), real_samples=real_samples)

                # Compute a mask to ignore padding
                mask = compute_mask(real_samples)
                # Compute loss: Mean Squared Error
                loss = criterion(mask.unsqueeze(-1) * outputs, real_samples)

                # Backward pass and optimize
                loss.backward()

                # Clip gradients to avoid exploding gradient problem
                if clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)

                opt.step()

                # Logging to TensorBoard
                writer.add_scalar("Loss/Gen", loss.item(), global_step=global_step)
                # Plot Gradients
                rnn_cell: nn.RNNCellBase
                writer.add_scalar("Grad/Gen/RNN", self.rnn_cell.weight_hh.grad.norm().item(),
                                  global_step=global_step)
                writer.add_scalar("Grad/Gen/Output", self.output[0].weight.grad.norm().item(),
                                  global_step=global_step)

                # Plotting
                if plot_freq > 0 and global_step % plot_freq == 0:
                    if notebook:
                        # Clear output and display plot
                        display.clear_output(wait=True)
                        # Display progressbars again
                        display.display(pbar_epochs.container)
                        display.display(pbar_batches.container)
                    if 'mnist' in dataset_name:
                        visualize_mnist_sequential(
                            gen_samples=outputs,
                            batches_done=global_step,
                            notebook=notebook,
                            tensorboard=tensorboard,
                            writer=writer,
                        )
                    elif dataset_name in list(Datasets):
                        # Trajectory Dataset
                        visualize_trajectory_samples(
                            gen_samples=outputs,
                            real_samples=real_samples,
                            real_lengths=lengths,
                            gen_lengths=lengths,
                            epoch=epoch,
                            batch_i=global_step,
                            notebook=notebook,
                            tensorboard=tensorboard,
                            writer=writer,
                        )

            # Save models per epoch
            if save_freq > 0 and epoch % save_freq == 0:
                self.save_parameters(epoch=epoch)

        writer.close()

    def predict_from_dataset(self, dataset: TrajectoryDataset, batch_size: int = 512) -> List[np.ndarray]:
        """Predict the same number of samples as in the dataset with the same length distribution."""
        # Create DataLoaders
        fs_test_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=ZeroPadding(return_len=True, return_labels=True)
        )
        # Predict
        predictions = []
        self.eval()
        with torch.no_grad():
            for batch in fs_test_dataloader:
                data, lengths, labels = split_data(batch)
                data.to(self.device)
                batch_size = len(data)
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                generated = self(noise, data.size(1)).detach().cpu().numpy()
                # Cut to length
                for i in range(batch_size):
                    predictions.append(generated[i, :lengths[i]])
        return predictions
