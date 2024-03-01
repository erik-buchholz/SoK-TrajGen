#!/usr/bin/env python3
"""
PyTorch Implementation of
C. Esteban, S. L. Hyland, and G. Rätsch,
“Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs.”
arXiv, Dec. 03, 2017. doi: 10.48550/arXiv.1706.02633.
Source (TensorFlow): https://github.com/ratschlab/RGAN/

"""
import logging
from contextlib import ExitStack
from datetime import datetime
from typing import Optional, Union, List

import torch
import torch.nn.functional as F
from IPython import display
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_ntbk

from stg.datasets.dataset_factory import Datasets
from stg.models.base_gan import BaseGAN
from stg.models.utils import get_rnn, get_optimizer, compute_wdiv_penalty, compute_gradient_penalty, contains_rnn, \
    clip_discriminator_weights, split_data, validate_loss_options, prepare_param_path, \
    Optimizer, NullWriter, visualize_trajectory_samples, visualize_mnist_sequential
from stg.utils.helpers import count_parameters_torch

log = logging.getLogger()


def get_rgan_name(GP: bool, WGAN: bool, N_CRITIC: int, OPT_D: str, OPT_G: str,
                  LR_D: float, LR_G: float, OUTPUT_DIM: int, dataset: Datasets or str,
                  WDIV: bool = False, LP: bool = False,
                  ):
    dataset_str = dataset.value if isinstance(dataset, Datasets) else dataset
    return f"RGAN_{'i' if GP else ''}{'W' if WGAN else ''}GAN{'-DIV' if WDIV else ''}" \
           f"{'-LP' if LP else ''}_G-{OPT_G}:{LR_G}_{N_CRITIC}x_D-{OPT_D}:{LR_D}_{dataset_str.upper()}:{OUTPUT_DIM}"


class RGANGenerator(nn.Module):
    def __init__(self,
                 noise_dim: int,
                 output_dim: int,
                 hidden_size: int,
                 rnn_type: str = 'lstm',
                 num_layers: int = 1,
                 dropout: float = 0
                 ):
        super().__init__()

        self.noise_dim = noise_dim
        self.rnn = get_rnn(rnn_type=rnn_type, input_size=noise_dim, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_dim, bias=True)

    def forward(self, x: torch.Tensor, lengths: List[int] = None) -> torch.Tensor:
        """
        Forward pass of the generator.
        :param x: Input shape = (batch_size, seq_len, self.noise_dim)
        :param lengths: Real lengths, used for padding
        :return: Output shape = (batch_size, seq_len, self.output_dim)
        """
        if lengths is not None:
            x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(x)
        if lengths is not None:
            y, output_lengths = pad_packed_sequence(y, batch_first=True)
        y = self.output_layer(y)
        return y.tanh()


class RGANDiscriminator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_size: int,
                 rnn_type: str = 'lstm',
                 num_layers: int = 1,
                 dropout: float = 0
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.rnn = get_rnn(rnn_type=rnn_type, input_size=input_dim, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=1, bias=True)

    def forward(self, x: torch.Tensor, lengths: List[int] = None) -> torch.Tensor:
        """
        :param x: Input shape = (batch_size, seq_len, self.input_dim)
        :param lengths: Real lengths, used for padding
        :return: Output shape = (batch_size, 1)
        """
        if lengths is not None:
            # Compute Mask and explicitly mask padding bits
            mask = torch.arange(x.size(1)).expand(len(lengths), x.size(1)).lt(
                torch.tensor(lengths).unsqueeze(-1)
            ).to(device=x.device)
            x = x * mask.unsqueeze(-1)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(x)
        if lengths is not None:
            y, output_lengths = pad_packed_sequence(y, batch_first=True)
        final_hidden_state = y[:, -1]
        y = self.output_layer(final_hidden_state)
        return y


class RGAN(BaseGAN):
    """Combined RGAN Model"""

    def __init__(
            self,
            gpu: int,
            noise_dim: int,
            output_dim: int,
            hidden_size: int,
            rnn_type: str = 'lstm',
            num_layers: int = 1,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)

        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu > -1 else "cpu")
        self.noise_dim = noise_dim

        self.gen = RGANGenerator(
            noise_dim=noise_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_layers
        ).to(device=device)
        self.dis = RGANDiscriminator(
            input_dim=output_dim,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_layers
        ).to(device=device)
        # Print Parameter counts for both models to compare strengths
        n_dis = count_parameters_torch(self.dis, False)
        n_gen = count_parameters_torch(self.gen, False)
        log.info(f"Discriminator Parameters:\t{n_dis:,}")
        log.info(f"Generator Parameters:\t{n_gen:,}")
        log.info(f"Proportion:\t\t\t{n_dis / n_gen:.2f}")

    def forward(self, x, lengths: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        return self.gen(x, lengths=lengths)

    def training_loop(
            self,
            dataloader: DataLoader,
            epochs: int,
            dataset_name: Datasets,
            name: Optional[str] = None,
            suffix: str = '',
            # Output and Tracking
            plot_freq: int = 200,
            save_freq: int = 100,
            param_path: Optional[str] = None,
            notebook: bool = False,
            tensorboard: bool = True,
            # Optimizer Options
            lr_d: float = 0.001,
            lr_g: float = None,
            opt_g: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            opt_d: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            beta1: float = 0.5,
            beta2: float = 0.999,
            # Loss function options
            wgan: bool = True,
            gradient_penalty: bool = True,
            wdiv: bool = False,
            lp: bool = False,
            clip_value: float = 0.01,
            n_critic: int = 1,
            lambda_gp: int = 10,
            padded_generator: bool = False,
    ) -> None:
        """
        Train a GAN.

        :param dataloader: DataLoader.
        :param epochs: Number of epochs to train
        :param dataset_name: Name of the dataset
        :param lr_g: Learning Rate of Generator
        :param lr_d: Learning Rate of Discriminator (default: None -> same as Generator LR)
        :param beta1: Only used for Adam (default: 0.5)
        :param beta2: Only used for Adam (default: 0.999)
        :param clip_value: Clipping value for Discriminator, only used for WGAN without Gradient Penalty. (default: 0.01)
        :param n_critic: How many times to train the discriminator per generator run? (default: 1)
        :param plot_freq: After how many BATCHES/STEPS should the model be evaluated? -1 to deactivate. (default: 200)
        :param save_freq: Number of EPOCHS to save model parameters. -1 to deactivate. (default: 100)
        :param param_path: Directory to contain model parameters. Files will be param_path/EPOCH_{GEN|DIS}.pth.
        :param notebook: Use the notebook version of TQDM and print to notebook instead of TensorBoard.
        :param wgan: Use WGAN loss function?
        :param gradient_penalty: Improved WGAN Loss function? (better performance, but more comp. expensive)
        :param wdiv: Use Wasserstein Divergence Loss?
        :param lp: Use Lipschitz Penalty?
        :param lambda_gp: Multiplier for Gradient Penalty (default: 10)
        :param opt_g: Optimizer to use for Generator. Default 'auto' uses recommended optimizer for loss function.
        :param opt_d: Optimizer to use for Discriminator. Default 'auto' uses recommended optimizer for loss function.
        :param name: Name of the model (for TensorBoard)
        :param suffix: Suffix for Model Name in Tensorboard
        :param padded_generator: Let the generator generate sequences of same length as input
        :param tensorboard: Use TensorBoard for logging?
        :return:
        """
        # Validate inputs
        validate_loss_options(wgan=wgan, gradient_penalty=gradient_penalty, lp=lp, wdiv=wdiv)

        # Move to device
        device = self.device

        # Determine Optimizer to use
        # Provided optimizers take precedence
        lr_g = lr_d if lr_g is None else lr_g
        if not isinstance(opt_g, torch.optim.Optimizer):
            opt_g = get_optimizer(parameters=self.gen.parameters(), choice=opt_g, lr=lr_g, beta_1=beta1, beta_2=beta2,
                                  wgan=wgan,
                                  gradient_penalty=gradient_penalty)
        if not isinstance(opt_d, torch.optim.Optimizer):
            opt_d = get_optimizer(parameters=self.dis.parameters(), choice=opt_d, lr=lr_d, beta_1=beta1, beta_2=beta2,
                                  wgan=wgan,
                                  gradient_penalty=gradient_penalty)

        # Get name for TensorBoard logging & parameter path
        if name is None:
            suffix = f"_{suffix}" if len(suffix) > 0 else suffix
            name = get_rgan_name(GP=gradient_penalty, WGAN=wgan, WDIV=wdiv, N_CRITIC=n_critic,
                                 OPT_D=type(opt_d).__name__,
                                 OPT_G=type(opt_g).__name__, LR_D=lr_d, LR_G=lr_g, OUTPUT_DIM=self.dis.input_dim,
                                 LP=lp, dataset=dataset_name) + suffix

        log.info(f"Model Name:\t\t{name}")

        # Determine Parameter Path and create directory
        self.param_path = prepare_param_path(name=name, param_path=param_path)

        # Create Tensorboard connection
        writer = SummaryWriter('runs/' + datetime.now().strftime('%b-%d_%X_') + name) if tensorboard else NullWriter()

        # Determine progressbar function
        tqdm_func = tqdm_ntbk if notebook else tqdm

        # Initialize counters
        batches_done = 0
        # Create progressbar for epochs
        pbar_epochs = tqdm_func(range(1, epochs + 1), desc="Epochs")
        for epoch in pbar_epochs:
            # Create progressbar for batches
            pbar_batches = tqdm_func(enumerate(dataloader), leave=False, total=len(dataloader), desc="Batches")
            for i, data in pbar_batches:

                samples, lengths, labels = split_data(data)
                gen_lengths = lengths if padded_generator else None

                # Ground Truth for Adversarial Loss (Standard GAN)
                valid = torch.ones(size=(samples.shape[0], 1), device=device) if not wgan else None
                fake = torch.zeros(size=(samples.shape[0], 1), device=device) if not wgan else None

                # Configure input
                real_samples = samples.to(device=device, dtype=torch.float32)
                if wdiv:
                    real_samples.requires_grad = True

                # ---------------------
                #  Train Discriminator
                # ---------------------

                opt_d.zero_grad()

                # Sample noise as generator input
                z = torch.randn(size=(real_samples.shape[0], real_samples.shape[1], self.gen.noise_dim), device=device)

                # Generate a batch of images
                gen_samples = self.gen(z, lengths=gen_lengths)

                # Adversarial loss
                gp = 0
                if wdiv:
                    # Wasserstein GAN with Wasserstein Divergence
                    with ExitStack() as stack:
                        if contains_rnn(self.dis):
                            # Disable cudnn for gradient computation if RNN is used
                            stack.enter_context(torch.backends.cudnn.flags(enabled=False))
                        real_validity = self.dis(real_samples, lengths=lengths)
                        fake_validity = self.dis(gen_samples, lengths=gen_lengths)
                    div_gp = compute_wdiv_penalty(
                        real=real_samples, fake=gen_samples,
                        real_validity=real_validity, fake_validity=fake_validity)
                    loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
                elif wgan:
                    # Wasserstein GAN
                    loss_D = -torch.mean(self.dis(real_samples, lengths=lengths)) + torch.mean(
                        self.dis(gen_samples, lengths=gen_lengths))
                    if gradient_penalty:
                        # With Gradient Penalty
                        gp = compute_gradient_penalty(discriminator=self.dis, real=real_samples, synthetic=gen_samples,
                                                      lengths=lengths, lp=lp)
                        loss_D += lambda_gp * gp
                else:
                    # Standard GAN
                    loss_D = (F.binary_cross_entropy_with_logits(self.dis(real_samples, lengths=lengths), valid) +
                              F.binary_cross_entropy_with_logits(self.dis(gen_samples, lengths=gen_lengths), fake)) / 2

                loss_D.backward()
                opt_d.step()

                # Clip weights of discriminator
                if wgan and not gradient_penalty:
                    clip_discriminator_weights(dis=self.dis, clip_value=clip_value)

                # Train the generator every n_critic iterations
                if i % n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    opt_g.zero_grad()

                    # Generate a batch of images
                    gen_samples = self.gen(z)

                    if wgan:
                        # Wasserstein GAN Loss
                        loss_G = -torch.mean(self.dis(gen_samples, lengths=gen_lengths))
                    else:
                        # Standard GAN Loss
                        loss_G = F.binary_cross_entropy_with_logits(self.dis(gen_samples, lengths=gen_lengths), valid)

                    loss_G.backward()
                    opt_g.step()

                    # Generator loss
                    writer.add_scalar("Loss/Gen", loss_G.item(), global_step=batches_done)

                    loss_D = loss_D.item()
                    if wgan:
                        if gradient_penalty:
                            # Remove gradient penalty and plot separately
                            loss_D -= lambda_gp * gp
                            writer.add_scalar("Loss/GP", gp, global_step=batches_done)

                        # For WGAN, one has to plot -D_loss not D_loss according to the authors:
                        # https://github.com/martinarjovsky/WassersteinGAN/issues/9#issuecomment-280989632
                        loss_D = -loss_D
                    writer.add_scalar("Loss/Dis", loss_D, global_step=batches_done)

                    # Plot gradients
                    writer.add_scalar("Grad/Gen/RNN", self.gen.rnn.weight_hh_l0.grad.norm(), global_step=batches_done)
                    writer.add_scalar("Grad/Gen/Lin", self.gen.output_layer.weight.grad.norm(),
                                      global_step=batches_done)
                    writer.add_scalar("Grad/Dis/RNN", self.dis.rnn.weight_hh_l0.grad.norm(), global_step=batches_done)
                    writer.add_scalar("Grad/Dis/Lin", self.dis.output_layer.weight.grad.norm(),
                                      global_step=batches_done)

                if plot_freq > 0 and batches_done % plot_freq == 0:
                    if notebook:
                        # Clear output and display plot
                        display.clear_output(wait=True)
                        # Display progressbars again
                        display.display(pbar_epochs.container)
                        display.display(pbar_batches.container)
                    if 'mnist' in dataset_name:
                        visualize_mnist_sequential(
                            gen_samples=gen_samples,
                            batches_done=batches_done,
                            notebook=notebook,
                            tensorboard=tensorboard,
                            writer=writer,
                        )
                    else:
                        # Trajectory Dataset
                        visualize_trajectory_samples(
                            gen_samples=gen_samples,
                            real_samples=real_samples,
                            real_lengths=lengths,
                            gen_lengths=gen_lengths,
                            epoch=epoch,
                            batch_i=batches_done,
                            notebook=notebook,
                            tensorboard=tensorboard,
                            writer=writer,
                        )

                batches_done += 1

            if (save_freq > 0 and epoch % save_freq == 0) or epoch == epochs:  # Always save after last epoch
                self.save_parameters(epoch=epoch)
