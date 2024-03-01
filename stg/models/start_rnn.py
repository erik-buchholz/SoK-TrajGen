#!/usr/bin/env python3
"""
This model is a simple RNN model that differs from the other models in that it receives the starting
point of the original sequence as guidance.
We noticed in previous experiments that the main issue appears to be the distribution of points.
Therefore, providing the starting point as guidance should help the model to learn the distribution
of points better.
Please note that this a relaxation from the privacy perspective as the start point might still, or especially
be private.
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


class StartRNN(BaseRNN):

    def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            hidden_size: int,
            num_layers: int = 1,
            dropout: float = 0.,
            rnn_type: utils.RnnType = utils.RnnType.LSTM,
            bidirectional: bool = False,
            output_func: nn.Module or None = nn.Tanh,
            output_dim: int = None,
            name: str = "StartRNN",
    ):
        super().__init__(
            name=name,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            output_func=output_func,
            output_dim=output_dim,
        )

        # Create layers
        self.embedding = nn.Linear(self.input_dim, self.embedding_dim)
        self.rnn = utils.get_rnn(rnn_type=self.rnn_type, input_size=self.embedding_dim, hidden_size=self.hidden_size,
                                 num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.output = nn.Linear(self.hidden_size, self.output_dim)
        if output_func is not None:
            self.output_func = output_func()

    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        # Embed input
        x = self.embedding(x)
        # Run through RNN
        out, state = self.rnn(x, h)

        out = self.output(out)
        if self.output_func is not None:
            out = self.output_func(out)
        return out, state

    def init_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        hidden = torch.zeros(self.num_layers * (1 + int(self.bidirectional)), batch_size, self.hidden_size,
                             device=device)
        if isinstance(self.rnn, nn.LSTM):
            cell = torch.zeros(self.num_layers * (1 + int(self.bidirectional)), batch_size, self.hidden_size,
                               device=device)
            return hidden, cell
        else:
            return hidden

    def predict_next(self, x: Tensor, h: Tensor = None) -> Tensor:
        """
        Predict the next point in the sequence.
        :param x: Current sequence
        :param h: Previous hidden state
        :return: Predicted sequence and hidden state.
        """
        out, state = self.forward(x, h)
        return out, state

    def generate(self, lengths: List[int], start_points: Tensor) -> Tensor:
        """
        Generate a sequence of given length.
        :param lengths: Length of the sequence to generate. Shape: (batch_size,)
        :param start_points: Starting points of the original sequences. Shape: (batch_size, input_dim)
        :return: Generated sequence
        """
        # Initialize hidden state
        batch_size = len(lengths)
        h = self.init_hidden(batch_size, self.device)
        # Initialize output
        out = torch.zeros(batch_size, max(lengths), self.input_dim, device=self.device)
        # Store starting points
        out[:, 0, :] = start_points
        # Initialize input: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = start_points.unsqueeze(1)

        # Eval Mode
        self.eval()

        # More efficient computation via not tracking gradients
        with torch.no_grad():
            # Generate sequence
            for i in range(1, max(lengths)):
                # Predict next point
                x, h = self.predict_next(x, h)
                # Store prediction
                out[:, i, :] = x.squeeze(1)

        return out

    def generate_from_test(self, test_dataset: TrajectoryDataset, batch_size: int = 512) -> List[np.ndarray]:

        # Create DataLoader
        fs_test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=ZeroPadding(return_len=True, return_labels=True)
        )

        predictions = []
        for batch in fs_test_dataloader:
            data, lengths, labels = split_data(batch)

            # Generate
            start_points = data[:, 0, :].to(device=self.device, dtype=torch.float32)
            out = self.generate(lengths, start_points)

            # Remove padding
            out = out.detach().cpu().numpy()
            for i, length in enumerate(lengths):
                predictions.append(out[i, :length, :])

        # Make sure generation worked as expected
        assert len(predictions) == len(test_dataset)
        # Make the first location of each trajectory the start point
        for i, (traj, _) in enumerate(test_dataset):
            assert np.allclose(predictions[i][0], traj[0]), f"Trajectory {i} does not start at the correct point"

        return predictions

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
                      suffix: str = "",
                      **kwargs
                      ) -> None:
        """
        Train the model.

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
            suffix = f"_{suffix}" if suffix != "" else suffix
            name = f"START_{self.rnn_type}_{dataset_name}_{lr}" + suffix
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

                data, lengths, _ = split_data(data)

                # Deal with datasets with constant lengths (lengths = None),\
                # i.e., all samples have the same maximum length
                if lengths is None:
                    lengths = [data.shape[1]] * data.shape[0]

                # Move to device
                data = data.to(device=device, dtype=torch.float32)

                # Input sequence used for training:
                # Each sequence is shifted by one to the left, i.e., the input sequence is the original sequence
                # without the last point. The expected output is the original sequence without the first point.
                input_seq = data[:, :-1, :]  # Shape: (batch_size, seq_len - 1, input_dim)
                real_samples = data  # Shape: (batch_size, seq_len, input_dim) (remember to add start point)

                # Reset gradients
                opt.zero_grad()

                # Forward pass
                outputs, _ = self.forward(input_seq)
                # Pre-pend start point to second dimension
                start_points = data[:, 0, :].unsqueeze(1)
                outputs = torch.cat((start_points, outputs), dim=1)


                # Compute a mask to ignore padding
                mask = compute_mask(real_samples)
                # Verify that mask is correct by comparing to lengths (note: lengths are without start point)
                lengths = torch.tensor(lengths)  # -1 if start point is not added
                assert torch.all(mask.sum(dim=1).cpu() == lengths), \
                    "Mask is incorrect: Sum of mask does not match lengths."
                # Compute loss: Mean Squared Error
                loss = criterion(mask.unsqueeze(-1) * outputs, real_samples)

                # Backward pass and optimize
                loss.backward()

                # Clip gradients to avoid exploding gradient problem
                if clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)

                # Update parameters
                opt.step()

                # Logging to TensorBoard
                writer.add_scalar("Loss/Gen", loss.item(), global_step=global_step)
                # Plot Gradients
                rnn_cell: nn.RNNCellBase
                writer.add_scalar("Grad/Gen/RNN", self.rnn.weight_hh_l0.grad.norm().item(),
                                  global_step=global_step)
                writer.add_scalar("Grad/Gen/Output", self.output.weight.grad.norm().item(),
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
