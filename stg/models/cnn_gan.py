#!/usr/bin/env python3
"""
GAN based on 1D Convolutions. Inspired by WaveGAN [1].
Implementation based on [2].

[1] C. Donahue, J. McAuley, and M. Puckette, “Adversarial Audio Synthesis.” arXiv, Feb. 08, 2019.
Accessed: May 18, 2023. [Online]. Available: https://arxiv.org/abs/1802.04208
[2] Mostafa ElAraby, “Pytorch Implementation of WaveGAN Model to Generate Audio,”
GitHub. Accessed: Nov. 11, 2023. [Online]. Available: https://github.com/mostafaelaraby/wavegan-pytorch

"""
import logging
from datetime import datetime
from timeit import default_timer as timer
from typing import Optional, Union, List

import torch
import torch.nn.functional as F
from IPython import display
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stg import config
from stg.datasets import Datasets
from stg.models import utils
from stg.models.base_gan import BaseGAN
from stg.models.utils import Optimizer
from stg.utils.helpers import count_parameters_torch

log = logging.getLogger()

# ### Constant ###
DEFAULT_NAME = "CNN_GAN"


# ### Utils ###
def compute_L_in(output_length, padding, output_padding, kernel_size, stride, upsample, depth) -> (int, int):
    """
    Computes the initial input length (L_in) for a given output length after passing through a specified
    number of transposed convolution layers. Adjusts the padding if L_in cannot be directly achieved.

    :param output_length: Desired output length after the transposed convolution.
    :param padding: Padding used in the transposed convolution.
    :param output_padding: Output padding used in the transposed convolution.
    :param kernel_size: Kernel size of the transposed convolution.
    :param stride: Stride of the transposed convolution.
    :param upsample: Upsample factor (if any, False otherwise).
    :param depth: Number of transposed convolution layers to be applied.
    :return: Tuple of (computed L_in, adjusted padding)
    """

    def layer_input_length(L_out, stride, kernel_size, padding, output_padding):
        return ((L_out - kernel_size - output_padding + 2 * padding) / stride) + 1

    L_in = output_length

    # Apply the calculation iteratively for each layer
    finished = False
    while not finished:
        finished = True
        if padding < 0:
            raise ValueError("Padding cannot be negative or larger 25 - no suitable value found!")
        for _ in range(depth):
            if upsample:
                # Adjust for upsampling
                L_in /= upsample
            L_in = layer_input_length(L_in, stride, kernel_size, padding, output_padding)
            if L_in % 1 != 0 or L_in < 0:
                # L_in is not feasible, adjust padding
                padding -= 1
                finished = False
                break  # Break out of for loop and restart with reduced padding

    return int(L_in), int(padding)


def compute_conv1_output_length(L_in, kernel_size, stride, padding, n_layers):
    """
    Computes the output length of a 1D convolutional layer given the input length and other parameters.
    :param L_in: Input length.
    :param kernel_size: Kernel size of the convolution.
    :param stride: Stride of the convolution.
    :param padding: Padding of the convolution.
    :param n_layers: Number of convolutional layers.
    :return: Output length.
    """
    L_out = L_in
    for _ in range(n_layers):
        L_out = ((L_out - kernel_size + 2 * padding) / stride) + 1
    return int(L_out)


# ### Layers ###
class Transpose1dLayer(nn.Module):
    """
        Author: Mostafa ElAraby
        Source: https://github.com/mostafaelaraby/wavegan-pytorch/blob/master/models.py
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int = 11,
            upsample: int = None,
            output_padding: int = 0,
            use_batch_norm: bool = False,
    ):
        """
        Constructor for the Transpose1dLayer class, a specialized layer for 1-dimensional
        transpose convolutional operations/

        :param in_channels: Number of channels in the input tensor.
        :param out_channels: Number of channels in the output tensor.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride of the convolution.
        :param padding: Padding added to both sides of the input. Defaults to 11.
        :param upsample: Optional upsampling factor for the input signal. If specified, upsampling
                         is applied before the convolution operation. Default is None.
        :param output_padding: Additional size added to one side of the output shape. Default is 1.
        :param use_batch_norm: If True, batch normalization is applied after convolution. Default is False.
        """
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample
        reflection_pad = nn.ConstantPad1d(kernel_size // 2, value=0)
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        conv1d.weight.data.normal_(0.0, 0.02)
        Conv1dTrans = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        batch_norm = nn.BatchNorm1d(out_channels)
        if self.upsample:
            operation_list = [reflection_pad, conv1d]
        else:
            operation_list = [Conv1dTrans]

        if use_batch_norm:
            operation_list.append(batch_norm)
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x):
        if self.upsample:
            # recommended by WaveGAN paper to use nearest upsampling
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        output = self.transpose_ops(x)
        return output


class Conv1D(nn.Module):
    """
        Author: Mostafa ElAraby
        Source: https://github.com/mostafaelaraby/wavegan-pytorch/blob/master/models.py
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_size: int,
            alpha: float = 0.2,
            stride: int = 4,
            padding: int = 11,
            use_batch_norm: bool = False,
            drop_prob: float = 0,
    ):
        """
        Constructor for the Conv1D class, which implements a 1-dimensional convolutional layer.

        :param input_channels: Number of channels in the input tensor (int).
        :param output_channels: Number of channels in the output tensor (int).
        :param kernel_size: Size of the convolution kernel (int).
        :param alpha: Negative slope coefficient for the Leaky ReLU activation function (float). Default: 0.2.
        :param stride: Stride of the convolution (int). Default: 4.
        :param padding: Zero-padding added to both sides of the input (int). Default: 11.
        :param use_batch_norm: Flag to determine the use of batch normalization (bool). Default: False.
        :param drop_prob: Dropout probability. If greater than 0, dropout is applied (float). Default: 0 = deactivated.
        """
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.alpha = alpha
        self.use_batch_norm = use_batch_norm
        self.use_drop = drop_prob > 0
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.conv1d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = F.leaky_relu(x, negative_slope=self.alpha)
        if self.use_drop:
            x = self.dropout(x)
        return x


# ### Models ###
class CNN_GAN_Generator(nn.Module):
    """
        Modified for purpose of this project.
        Author of original version: Mostafa ElAraby
        Source: https://github.com/mostafaelaraby/wavegan-pytorch/blob/master/models.py
    """

    def __init__(
            self,
            output_length: int,
            output_dim: int = 1,
            noise_dim: int = 100,
            use_batch_norm: bool = True,
            upsample: bool = False,
    ):
        """
        Constructor for the Generator class, which implements a 1-dimensional convolutional generator.
        :param output_length: Length of the output signal.
        :param output_dim: Number of channels in the output signal.
        :param noise_dim: Dimension of the noise vector serving as input.
        :param use_batch_norm: Flag to determine the use of batch normalization (bool). Default: False.
        :param upsample: Flag to determine the use of upsampling (bool). Default: False.
        """
        super().__init__()
        # Store Parameters

        self.output_length = output_length
        self.noise_dim = noise_dim  # Dimension of the noise vector serving as input.
        # Expected input is z.shape = (batch_size, noise_dim)
        self.output_dim = output_dim  # num_channels in WaveGAN implementation
        self.use_batch_norm = use_batch_norm

        self.dim_mul = 2  # Multiplier for the number of channels in each layer

        # Compute required model size to reach output dim
        # output_dim should be approx. (self.dim_mul * model_size) // 32
        # to approx. half the # channels in each step
        model_size = (output_dim * 32) // self.dim_mul
        self.model_size = model_size
        log.debug(f"Generator Model size: {model_size}")

        stride = 1  # WaveGAN uses 4
        # TODO: For MNIST we cannot use larger stride, but for other datasets we could use at least 2
        if upsample:
            stride = 1
            upsample = 4  # WaveGAN uses 4
        kernel_size = 25  # WaveGAN uses 25
        padding = 11  # WaveGAN uses 11
        output_padding = 0  # WaveGAN uses 1, but we don't want to have an additional location at the end

        factor = upsample if upsample else 1
        self.start_length, padding = compute_L_in(
            output_length, padding, output_padding, kernel_size, stride, factor, 5)
        log.debug(f"Start length: {self.start_length}, padding: {padding}")

        self.fc1 = nn.Linear(noise_dim, self.start_length * model_size * self.dim_mul)
        self.bn1 = nn.BatchNorm1d(num_features=model_size * self.dim_mul)

        deconv_layers = [Transpose1dLayer(
            self.dim_mul * model_size,
            (self.dim_mul * model_size) // 2,
            kernel_size,
            stride,
            upsample=upsample,
            use_batch_norm=use_batch_norm,
            padding=padding,
            output_padding=output_padding,
        ), Transpose1dLayer(
            (self.dim_mul * model_size) // 2,
            (self.dim_mul * model_size) // 4,
            kernel_size,
            stride,
            upsample=upsample,
            use_batch_norm=use_batch_norm,
            padding=padding,
            output_padding=output_padding,
        ), Transpose1dLayer(
            (self.dim_mul * model_size) // 4,
            (self.dim_mul * model_size) // 8,
            kernel_size,
            stride,
            upsample=upsample,
            use_batch_norm=use_batch_norm,
            padding=padding,
            output_padding=output_padding,
        ), Transpose1dLayer(
            (self.dim_mul * model_size) // 8,
            (self.dim_mul * model_size) // 16,
            kernel_size,
            stride,
            upsample=upsample,
            use_batch_norm=use_batch_norm,
            padding=padding,
            output_padding=output_padding,
        ), Transpose1dLayer(
            (self.dim_mul * model_size) // 16,
            output_dim,
            kernel_size,
            stride,
            upsample=upsample,
            padding=padding,
            output_padding=output_padding,
        )]

        self.deconv_list = nn.ModuleList(deconv_layers)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x: Tensor):
        """
        Forward pass of the generator.
        :param x: Expected shape: (batch_size, self.noise_dim)
        :return: Output of the generator. Shape: (batch_size, self.output_length, self.output_dim)
        """
        log.debug(f"Generator input shape: {x.shape}")
        x = self.fc1(x).view(-1, self.dim_mul * self.model_size, self.start_length)
        log.debug(f"Generator fc1 output shape: {x.shape}")
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)

        for i, deconv in enumerate(self.deconv_list[:-1]):
            x = F.relu(deconv(x))
            log.debug(f"Generator deconv{i} output shape: {x.shape}")

        output = torch.tanh(self.deconv_list[-1](x))
        log.debug(f"Generator deconv{len(self.deconv_list) - 1} output shape: {output.shape}")
        output = output.permute(0, 2, 1)
        log.debug(f"Generator output shape: {output.shape}")
        # Before: (batch_size, output_dim = channels, output_length) # After: (batch_size, output_length, output_dim)
        return output


class CNN_GAN_Discriminator(nn.Module):
    """
        Modified for purpose of this project.
        Author of original version: Mostafa ElAraby
        Source: https://github.com/mostafaelaraby/wavegan-pytorch/blob/master/models.py
    """

    def __init__(
            self,
            input_length: int,
            input_dim: int = 1,
            use_batch_norm: bool = True,
            alpha: float = 0.2,

    ):
        """
        Constructor for the Discriminator class, which implements a 1-dimensional convolutional discriminator.
        :param input_length: Length of the input signal.
        :param input_dim: Number of channels in the input signal.
        :param use_batch_norm: Flag to determine the use of batch normalization (bool). Default: False.
        :param alpha: Negative slope coefficient for the Leaky ReLU activation function (float). Default: 0.2.
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.input_dim = input_dim  # num_channels in WaveGAN implementation
        self.alpha = alpha

        model_size = input_dim
        log.debug(f"Discriminator Model size: {model_size}")
        stride = 1  # WaveGAN uses 4
        padding = 11  # WaveGAN uses 11
        kernel_size = 25  # WaveGAN uses 25

        conv_layers = [
            Conv1D(
                input_dim,
                model_size,
                kernel_size,
                stride=stride,
                padding=padding,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
            ),
            Conv1D(
                model_size,
                2 * model_size,
                kernel_size,
                stride=stride,
                padding=padding,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
            ),
            Conv1D(
                2 * model_size,
                4 * model_size,
                kernel_size,
                stride=stride,
                padding=padding,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
            ),
            Conv1D(
                4 * model_size,
                8 * model_size,
                kernel_size,
                stride=stride,
                padding=padding,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
            ),
            Conv1D(
                8 * model_size,
                16 * model_size,
                kernel_size,
                stride=stride,
                padding=padding,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
            ),
            Conv1D(
                16 * model_size,
                32 * model_size,
                kernel_size,
                stride=stride,
                padding=padding,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
            )
        ]

        num_final_channels = 32 * model_size
        final_length = compute_conv1_output_length(input_length, kernel_size, stride, padding, len(conv_layers))

        self.fc_input_size = num_final_channels * final_length

        self.conv_layers = nn.ModuleList(conv_layers)

        self.fc1 = nn.Linear(self.fc_input_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    # noinspection PyUnusedLocal
    def forward(self, x, lengths=None):
        """
        Forward pass of the discriminator.
        :param x: Expected shape: (batch_size, self.input_length, self.input_dim)
        :param lengths: For compatibility with LSTM-based models
        :return:
        """
        batch_size = x.shape[0]
        log.debug(f"Discriminator input shape: {x.shape}")
        x = x.permute(0, 2, 1)
        log.debug(f"Discriminator permuted input shape: {x.shape}")
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            log.debug(f"Discriminator conv{i} output shape: {x.shape}")
        x = x.view(batch_size, self.fc_input_size)
        log.debug(f"Discriminator reshaped output shape: {x.shape}")
        x = self.fc1(x)
        log.debug(f"Discriminator fc1 output shape: {x.shape}")
        return x


class CNN_GAN(BaseGAN):

    def __init__(
            self,
            gpu: int,
            output_length: int,
            output_dim: int,
            noise_dim: int = 100,
            use_batch_norm: bool = True,
            upsample: bool = False,
            alpha: float = 0.2,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.noise_dim = noise_dim
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu > -1 else "cpu")

        self.gen = CNN_GAN_Generator(
            output_length=output_length,
            output_dim=output_dim,
            noise_dim=noise_dim,
            use_batch_norm=use_batch_norm,
            upsample=upsample
        ).to(device=device)
        self.dis = CNN_GAN_Discriminator(
            input_length=output_length,
            input_dim=output_dim,
            use_batch_norm=use_batch_norm,
            alpha=alpha
        ).to(device=device)

        # Print Parameter counts for both models to compare strengths
        n_dis = count_parameters_torch(self.dis, False)
        n_gen = count_parameters_torch(self.gen, False)
        log.info(f"Discriminator Parameters:\t{n_dis:,}")
        log.info(f"Generator Parameters:\t{n_gen:,}")
        log.info(f"Proportion:\t\t\t{n_dis / n_gen:.2f}")

    def forward(self, x, lengths: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        return self.gen(x)

    def training_loop(
            self,
            dataloader: DataLoader,
            epochs: int,
            dataset_name: Datasets,
            # Output and Tracking
            save_freq: int = 10,
            plot_freq: int = 100,
            tensorboard: bool = True,
            notebook: bool = False,
            # Optimizer Options
            lr_g: float = 1e-4,
            lr_d: float = 3e-4,
            opt_g: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            opt_d: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            beta1: float = 0.5,
            beta2: float = 0.999,
            # Loss function options
            wgan: bool = False,
            gradient_penalty: bool = False,
            lp: bool = False,
            n_critic: int = 1,
            clip_value: float = 0.01,
            lambda_gp: float = 10,
    ):
        """
        Training loop for the GAN.
        :param dataloader: DataLoader for the dataset.
        :param epochs: Number of epochs to train.
        :param dataset_name: Name of the dataset.
        :param save_freq: After how many epochs to save the current state.
        :param plot_freq: After how many batches to plot the current state.
        :param tensorboard: Track training progress with TensorBoard.
        :param notebook: Run in Jupyter notebook mode.
        :param lr_g: Learning rate for the generator.
        :param lr_d: Learning rate for the discriminator.
        :param opt_g: Optimizer for the generator.
        :param opt_d: Optimizer for the discriminator.
        :param beta1: Beta1 for the Adam optimizer.
        :param beta2: Beta2 for the Adam optimizer.
        :param wgan: Use (improved) Wasserstein GAN loss.
        :param gradient_penalty: Use gradient penalty for WGAN loss.
        :param lp: Use Lipschitz penalty instead of gradient penalty.
        :param n_critic: Number of discriminator runs per generator run.
        :param clip_value: Clipping value for the discriminator weights. Only used if WGAN is used without gradient.
        :param lambda_gp: Weight factor for the gradient penalty.
        :return:
        """
        # Validity checks
        utils.validate_loss_options(wgan=wgan, gradient_penalty=gradient_penalty, lp=lp)
        if wgan and n_critic <= 1 and lr_d <= lr_g:
            # If the discriminator LR is larger than the generator LR, this mimics n_critic > 1
            log.warning(f"Are you sure you want to use WGAN with even runs for Dis and Gen?")

        # Optimizers
        self.opt_g = utils.get_optimizer(
            parameters=self.gen.parameters(), choice=opt_g, lr=lr_g, beta_1=beta1, beta_2=beta2, wgan=wgan,
            gradient_penalty=gradient_penalty)
        log.info(
            f"Generator Optimizer:\t\t{self.opt_g.__class__.__name__} with lr={lr_g}, beta1={beta1}, and beta2={beta2}."
        )
        self.opt_d = utils.get_optimizer(
            parameters=self.dis.parameters(), choice=opt_d, lr=lr_d, beta_1=beta1, beta_2=beta2, wgan=wgan,
            gradient_penalty=gradient_penalty)
        log.info(
            f"Discriminator Optimizer:\t{self.opt_d.__class__.__name__} with lr={lr_d}, beta1={beta1}, and beta2={beta2}."
        )

        # Change name if none
        if self.name is None:
            self.name = DEFAULT_NAME
            # Append most important parameters to name for better identification
            self.name += (
                f"_{'WGAN' if wgan else 'GAN'}{'-GP' if gradient_penalty and not lp else ''}"
                f"{'-LP' if lp else ''}_G-{self.opt_g.__class__.__name__}:{lr_g}_{n_critic}xD-"
                f"{self.opt_d.__class__.__name__}:{lr_d}_{dataset_name.upper()}")
        log.info(f"Model name:\t\t\t{self.name}")

        # Update Parameter path
        self.param_path = utils.prepare_param_path(self.name)

        # Update Parameter path
        self.param_path = utils.prepare_param_path(name=self.name)

        # Import either notebook or normal tqdm
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        # Prepare Models
        self.dis.train()
        self.gen.train()

        # Create Tensorboard connection
        writer = SummaryWriter(config.BASE_DIR + 'runs/' + datetime.now().strftime('%b-%d_%X_') +
                               self.name) if tensorboard else utils.NullWriter()

        pbar_epochs = tqdm(range(1, epochs + 1), desc="Epochs")
        d_steps, g_steps, batches_done = 0, 0, 1
        for epoch in pbar_epochs:
            pbar_batches = tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Batches")
            for i, data in pbar_batches:
                batch_start = timer()

                real, lengths, labels = utils.split_data(data)

                # Configure input
                real = real.to(device=self.device, dtype=torch.float32)
                batch_size = real.shape[0]

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.dis.train()
                self.opt_d.zero_grad()

                # Generate a batch of synthetic trajectories
                # (For training of discriminator --> Generator in Eval mode)
                self.gen.eval()  # Need to be set back to training mode later
                noise = torch.randn(size=(batch_size, self.noise_dim), device=self.device)
                generated = self.gen(noise).detach()

                gp = -1  # Dummy value to avoid warnings later
                if wgan:
                    # (Improved) Wasserstein GAN
                    d_real = torch.mean(self.dis(real))
                    d_fake = torch.mean(self.dis(generated))
                    d_loss = -d_real + d_fake  # Vanilla WGAN loss
                    if gradient_penalty:
                        gp = utils.compute_gradient_penalty(
                            self.dis, real=real, synthetic=generated, lengths=lengths, lp=lp)
                        d_loss += lambda_gp * gp
                else:
                    # Discriminator Ground Truth (real=0, fake=1)
                    real_labels = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
                    syn_labels = torch.ones((batch_size, 1), device=self.device, dtype=torch.float32)
                    # BCE loss: By using with logits, we don't need to apply sigmoid to the output of the discriminator
                    # with allows for interoperability with other losses (e.g. WGAN)
                    real_loss = F.binary_cross_entropy_with_logits(self.dis(real), real_labels)
                    fake_loss = F.binary_cross_entropy_with_logits(self.dis(generated), syn_labels)
                    d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.opt_d.step()
                d_steps += 1

                # Only if WGAN w/o gradient penalty used
                if wgan and not gradient_penalty:
                    utils.clip_discriminator_weights(dis=self.dis, clip_value=clip_value)

                g_loss = None
                if batches_done % n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    self.gen.train()  # Switch training mode on for generator
                    self.opt_g.zero_grad()

                    # Sample noise
                    noise = torch.randn(size=(batch_size, self.noise_dim), device=self.device)

                    generated = self.gen(noise)

                    if wgan:
                        g_loss = -torch.mean(self.dis(generated))
                    else:
                        # Create proper label in case discriminator's labels are noisy
                        real_labels = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
                        g_loss = F.binary_cross_entropy_with_logits(self.dis(generated), real_labels)

                    g_loss.backward()
                    self.opt_g.step()
                    g_steps += 1

                # Generator loss
                if g_loss is not None:
                    writer.add_scalar("Loss/Gen", g_loss.item(), global_step=batches_done)
                # Discriminator loss
                d_loss = d_loss.item()
                if wgan:
                    if gradient_penalty:
                        # Remove gradient penalty and plot separately
                        d_loss -= lambda_gp * gp
                        writer.add_scalar("Loss/GP", gp, global_step=batches_done)
                    # For WGAN, one has to plot -D_loss not D_loss according to the authors.
                    d_loss = -d_loss
                writer.add_scalar("Loss/Dis", d_loss, global_step=batches_done)

                # Plot trajectories
                if plot_freq > 0 and batches_done % plot_freq == 0:
                    if notebook:
                        # Clear output and display plot
                        display.clear_output(wait=True)
                        # Display progressbars again
                        display.display(pbar_epochs.container)
                        display.display(pbar_batches.container)
                    if 'mnist' in dataset_name:
                        utils.visualize_mnist_sequential(
                            gen_samples=generated,
                            batches_done=batches_done,
                            notebook=notebook,
                            tensorboard=tensorboard,
                            writer=writer,
                        )
                    else:
                        # Trajectory Dataset
                        utils.visualize_trajectory_samples(
                            gen_samples=generated,
                            real_samples=real,
                            real_lengths=lengths,
                            gen_lengths=lengths,
                            epoch=epoch,
                            batch_i=batches_done,
                            notebook=notebook,
                            tensorboard=tensorboard,
                            writer=writer,
                        )

                batches_done += 1
                log.debug(f"Batch completed in {timer() - batch_start:.2f}s")
                # End Batch

            if save_freq > 0 and epoch % save_freq == 0:
                self.save_parameters(epoch)
            # End Epoch

        print(f"Total Generator steps: {g_steps}; Total Discriminator Steps: {d_steps}")
