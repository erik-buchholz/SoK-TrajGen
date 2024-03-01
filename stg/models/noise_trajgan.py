#!/usr/bin/env python3
"""Implementation of the Noise-TrajGAN model.
Noise-TrajGAN is based on LSTM-TrajGAN, however, in contrast to LSTM-TrajGAN, Noise-TrajGAN only
receives noise as an input to prevent any leakage to the generator's output.
Other than this modification, we tried to keep the code as close to the original as possible.
"""
import logging
import warnings
from contextlib import nullcontext
from datetime import datetime
from typing import List, Dict, Optional

import math
import torch
from IPython import display
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer

from stg import config
from stg.datasets import Datasets
from stg.models import utils
from stg.models.lstm_trajgan import LSTM_TrajGAN_FEATURES, VOCAB_SIZE, EMBEDDING_SIZE
from stg.models.trajGAN import TrajGAN
from stg.models.traj_loss import get_trajLoss
from stg.models.utils import Optimizer, validate_loss_options, get_optimizer, NullWriter, split_data, \
    compute_gradient_penalty, l1_regularizer, clip_discriminator_weights

# CONSTANTS ####################################################################
log = logging.getLogger()
LEARNING_RATE = 0.005
BETA1 = 0.5
BATCH_SIZE = 256
MAX_PHYSICAL_BATCH_SIZE = 3000
LAMBDA_GP = 10


class Generator(nn.Module):
    def __init__(
            self,
            features: List[str],
            vocab_size: Dict[str, int],
            embedding_size: Dict[str, int],
            noise_dim: int,
            latent_dim: int = 100,
            recurrent_layers: int = 1,
            dp: bool = False,
            traj_shaped_noise=False,
    ):
        super().__init__()

        # Store (hyper)parameters
        self.features = features
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.noise_dim = noise_dim
        self.recurrent_layers = recurrent_layers
        self.dp = dp
        self.traj_shaped_noise = traj_shaped_noise

        # Create Model

        # Embedding Layer (although the input is noise, so this is not really an embedding)
        if self.traj_shaped_noise:
            self.embedding_layers = nn.ModuleDict()
            for i, feature in enumerate(self.features):
                self.embedding_layers[feature] = nn.Sequential(
                    nn.Linear(vocab_size[feature], embedding_size[feature], bias=True, dtype=torch.float32),
                    nn.ReLU()
                )
            feature_len = sum(self.embedding_size[f] for f in self.features)
        else:
            feature_len = self.noise_dim  # Used for noise of shape (batch_size, time_steps, latent_dim)

        # Feature Fusion
        self.feature_fusion = nn.Sequential(nn.Linear(feature_len, latent_dim, bias=True, dtype=torch.float32),
                                            nn.ReLU())

        # LSTM Layer
        if dp:
            self.lstm = DPLSTM(latent_dim, latent_dim, batch_first=True, num_layers=self.recurrent_layers)
        else:
            self.lstm = nn.LSTM(latent_dim, latent_dim, batch_first=True, dtype=torch.float32,
                                num_layers=self.recurrent_layers)

        # Output Layer
        output_latlon = nn.Sequential(
            nn.Linear(latent_dim, self.vocab_size[self.features[0]], bias=True, dtype=torch.float32),
            nn.Tanh()
        )

        # We expect latlon to be the minimal output
        self.output_layers = nn.ModuleDict({
            'latlon': output_latlon,
        })
        for feature in self.features[1:]:
            self.output_layers[feature] = nn.Sequential(
                nn.Linear(latent_dim, self.vocab_size[feature], bias=True, dtype=torch.float32),
                nn.Softmax(dim=-1)
            )

    def forward(self, x: Tensor):
        """

        :param x: Noise (Tensor) w/ shape (batch_size, latent_dim)
        :return: outputs.shape = (num_features, batch_size, time_steps, feature_size)
        """
        if self.traj_shaped_noise:
            # Embedding Layer
            embeddings = []
            for i, feature in enumerate(self.features):
                embeddings.append(self.embedding_layers[feature](x[i].to(dtype=torch.float32)))
            noise = torch.cat(embeddings, dim=-1)
        else:
            # Noise provided in shape (batch_size, time_steps, latent_dim)
            noise = x

        # Feature Fusion
        fusion = self.feature_fusion(noise)
        # LSTM Layer
        lstm, _ = self.lstm(fusion)
        # Output Layer
        latlon = self.output_layers['latlon'](lstm)
        outputs = [latlon, ]
        for feature in self.features[1:]:
            outputs.append(self.output_layers[feature](lstm))

        return tuple(outputs)


class Discriminator(nn.Module):
    def __init__(
            self,
            features: List[str],
            vocab_size: Dict[str, int],
            embedding_size: Dict[str, int],
            latent_dim: int,
            dp: bool = False,
    ):
        super().__init__()

        # Store (hyper)parameters
        self.features = features
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dp = dp

        # Create Model
        self.embedding_layers = nn.ModuleDict()
        for i, feature in enumerate(self.features):
            self.embedding_layers[feature] = nn.Sequential(
                nn.Linear(vocab_size[feature], embedding_size[feature], bias=True, dtype=torch.float32),
                nn.ReLU()
            )
        feature_len = sum(self.embedding_size[f] for f in self.features)
        self.feature_fusion = nn.Sequential(nn.Linear(feature_len, latent_dim, dtype=torch.float32), nn.ReLU())
        if dp:
            self.lstm = DPLSTM(latent_dim, latent_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(latent_dim, latent_dim, batch_first=True, dtype=torch.float32)
        self.output_layer = nn.Sequential(
            nn.Linear(latent_dim, 1, bias=True, dtype=torch.float32),
        )

    def forward(self, x):
        # Embedding Layer
        embeddings = []
        for i, feature in enumerate(self.features):
            embeddings.append(self.embedding_layers[feature](x[i].to(dtype=torch.float32)))
        # Feature Fusion
        concat = torch.cat(embeddings, dim=-1)
        fusion = self.feature_fusion(concat)
        # LSTM Layer
        _, (final_hidden_state, _) = self.lstm(fusion)
        # Output Layer
        validity = self.output_layer(final_hidden_state[-1])
        return validity


class Noise_TrajGAN(TrajGAN):
    def __init__(
            self,
            # Architecture Options
            features: List[str] = LSTM_TrajGAN_FEATURES,
            vocab_size: Dict[str, int] = VOCAB_SIZE,
            embedding_size: Dict[str, int] = EMBEDDING_SIZE,
            latent_dim: int = 100,
            noise_dim: int = 100,
            recurrent_layers: int = 1,
            use_regularizer: bool = True,
            traj_shaped_noise: bool = False,
            # General Options
            param_path: Optional[str] = None,
            gpu: Optional[int] = None,
            name: Optional[str] = None,
            # Optimizer Options
            lr_g: float = LEARNING_RATE,
            lr_d: float = LEARNING_RATE,
            opt_g: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            opt_d: torch.optim.Optimizer or Optimizer = Optimizer.AUTO,
            beta1: float = BETA1,
            beta2: float = 0.999,
            # GAN Loss options
            wgan: bool = True,
            gradient_penalty: bool = True,
            lipschitz_penalty: bool = False,
            # Privacy Options
            dp: bool = False,
            privacy_accountant: str = "rdp",
            epsilon: int = 10,
            delta: float = 1e-5,
    ):
        """
            Initialize the Noise-TrajGAN model with the specified architecture and optimization settings.

            :param features: A list of feature names to be used in the GAN.
            :param vocab_size: A dictionary mapping feature names to their vocabulary sizes.
            :param embedding_size: A dictionary mapping feature names to their embedding sizes.
            :param latent_dim: The dimensionality of the latent space.
            :param noise_dim: The dimensionality of the noise vector.
            :param recurrent_layers: The number of recurrent layers to use in the model.
            :param use_regularizer: Flag indicating whether to use a L1 regularizer.
            :param traj_shaped_noise: Flag indicating whether noise should be shaped like trajectories.
            :param param_path: The path to save or load model parameters.
            :param name: The name of the model for identification purposes.
            :param gpu: The GPU device ID to use. If `None`, CPU is used.
            :param lr_g: The learning rate for the generator.
            :param lr_d: The learning rate for the discriminator.
            :param opt_g: The optimizer to use for the generator.
            :param opt_d: The optimizer to use for the discriminator.
            :param beta1: The beta1 hyperparameter for Adam optimizer.
            :param beta2: The beta2 hyperparameter for Adam optimizer.
            :param dp: Flag indicating whether differential privacy should be used.
            :param privacy_accountant: The type of privacy accountant to use for differential privacy.
            :param wgan: Flag indicating whether to use the Wasserstein GAN formulation.
            :param gradient_penalty: Flag indicating whether to apply gradient penalty in GAN loss.
            :param lipschitz_penalty: Flag indicating whether to apply Lipschitz penalty in GAN loss.

            :return: An instance of the Noise_TrajGAN model.
            """
        super().__init__()

        # Store Data
        self.features = features
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.use_regularizer = use_regularizer
        self.dp = dp
        self.traj_shaped_noise = traj_shaped_noise
        self.wgan = wgan
        self.gradient_penalty = gradient_penalty
        self.lipschitz_penalty = lipschitz_penalty
        self.epsilon = epsilon
        self.delta = delta

        # Validate input
        validate_loss_options(wgan=wgan, gradient_penalty=gradient_penalty, lp=lipschitz_penalty)

        if gpu is not None and gpu > -1 and torch.cuda.is_available():
            device = f"cuda:{gpu}"
        else:
            device = "cpu"

        # Create components
        self.gen = Generator(
            features=features,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            noise_dim=noise_dim,
            latent_dim=latent_dim,
            recurrent_layers=recurrent_layers,
            dp=self.dp,
            traj_shaped_noise=traj_shaped_noise
        ).to(device)
        self.dis = Discriminator(
            features=features,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            latent_dim=latent_dim,
            dp=self.dp,
        ).to(device)

        # Loss functions for standard GAN
        self.dis_loss = torch.nn.BCEWithLogitsLoss()
        # Adapt TrajLoss to provided features
        self.gen_loss = get_trajLoss(features=self.features, weight=1)

        # Determine Optimizers
        if not isinstance(opt_g, torch.optim.Optimizer):
            self.opt_g = get_optimizer(parameters=self.gen.parameters(), choice=opt_g, lr=lr_g, beta_1=beta1,
                                       beta_2=beta2, wgan=self.wgan, gradient_penalty=self.gradient_penalty)
        else:
            self.opt_g = opt_g
        if not isinstance(opt_d, torch.optim.Optimizer):
            self.opt_d = get_optimizer(parameters=self.dis.parameters(), choice=opt_d, lr=lr_d, beta_1=beta1,
                                       beta_2=beta2, wgan=self.wgan, gradient_penalty=self.gradient_penalty)
        else:
            self.opt_d = opt_d

        # Determine Model Name
        if name is None:
            name = "Noise_TrajGAN"
        if self.dp and 'dp' not in name.lower():
            # Add DP to name
            name = "DP_" + name
        self.name = name

        # Configure and Create Parameter paths
        self.prepare_param_paths(name=self.name, param_path=param_path)

        # -------------------------------
        #  Differential Private Stochastic Gradient Descent
        # Validate if model can be used for DP-SGD without modifications
        if self.dp:
            from opacus.validators import ModuleValidator
            errors = ModuleValidator.validate(self, strict=False)
            if len(errors) == 0:
                log.info("No errors found - model can be trained with DP.")
            else:
                log.error("The following errors prevent the model from DP training:")
                for e in errors:
                    log.error(str(e))
                    ModuleValidator.validate(self, strict=True)
            self.accountant = privacy_accountant
            self.dp_initialised = False
            with warnings.catch_warnings(record=True) as w:
                self.privacy_engine = PrivacyEngine(accountant=self.accountant)
                if len(w) > 0:
                    for warning in w:
                        log.warning(str(warning.message))
        # -------------------------------

    def forward(self, x):
        return self.gen(x)

    def get_noise(self,
                  real_trajs: List[Tensor] = None,
                  batch_size: int = None,
                  num_time_steps: int = None,
                  feature_dims: List[int] = None,
                  traj_shaped_noise: bool = False) -> Tensor:
        """
        Real Trajectories are only used for the shape of the noise, no information is leaked!
        :param real_trajs:  List of real trajectories for shape
        :param batch_size: Batch size
        :param num_time_steps: Number of time steps
        :param feature_dims: Feature dimensions
        :param traj_shaped_noise: Whether to use noise with shape (n_feature, batch_size, num_time_steps, feature_dim)
        :return:
        """
        # Check that either real_trajs or num_features, batch_size and num_time_steps are provided
        if real_trajs is None and (batch_size is None or num_time_steps is None):
            raise ValueError("Either provide real_trajs or batch_size and num_time_steps!")
        if real_trajs is not None and (batch_size is not None or num_time_steps is not None):
            raise ValueError("Provide either real_trajs or batch_size and num_time_steps!")
        if traj_shaped_noise and real_trajs is None and feature_dims is None:
            raise ValueError("Provide feature_dims or real_trajs if traj_shaped_noise is True!")

        if real_trajs is not None:
            batch_size = len(real_trajs[0])
            num_time_steps = len(real_trajs[0][0])
            feature_dims = [x.shape[-1] for x in real_trajs]
        if not traj_shaped_noise:
            noise = torch.randn(size=(batch_size, num_time_steps, self.noise_dim), device=self.device)
        else:
            noise = [torch.randn(size=(batch_size, num_time_steps, dim), device=self.device) for i, dim in
                     enumerate(feature_dims)]
        return noise

    def training_loop(self,
                      dataloader: DataLoader,
                      epochs: int,
                      dataset_name: Datasets,
                      save_freq: int = 10,  # Save every x epochs
                      plot_freq: int = 100,  # Plot every x batches
                      n_generator: int = 1,  # Number of generator runs per batch
                      n_critic: int = 1,  # Number of discriminator runs per Generator run
                      clip_value: float = 0.01,  # Clip discriminator weights
                      tensorboard: bool = True,
                      lambda_gp: int = LAMBDA_GP,
                      notebook: bool = False,  # Run in Jupyter Notebook
                      ) -> None:
        """

        :param dataloader: DataLoader
        :param epochs: Number of epochs to train
        :param dataset_name: Name of the dataset
        :param save_freq: Save every x epochs. -1 to disable saving. (Default: 10)
        :param plot_freq: Plot every x batches. -1 to disable plotting. (Default: 100)
        :param n_generator: Number of generator runs per batch. (Default: 1)
        :param n_critic: Number of discriminator runs per Generator run. (Default: 1)
        :param clip_value: Clip discriminator weights in case of WGAN. (Default: 0.01)
        :param tensorboard: Enable Tensorboard logging. (Default: True)
        :param lambda_gp: Gradient penalty coefficient for iWGAN/WGAN-GP/WGAN-LP. (Default: 10)
        :param notebook: Run in Jupyter Notebook. (Default: False)
        :return:
        """

        # Ignore the two backward hook warnings as Opacus confirmed that these
        # can safely be ignored
        warnings.filterwarnings("ignore", message=".* non-full backward hook")

        # Validity checks
        validate_loss_options(wgan=self.wgan, gradient_penalty=self.gradient_penalty, lp=self.lipschitz_penalty)
        if self.wgan and n_generator == n_critic:
            log.warning(f"Are you sure you want to use WGAN with even runs for Dis and Gen?")
        if self.dp and not self.dp_initialised:
            raise RuntimeError("Call .dp_init() before training!")
        if self.dp and self.epochs != epochs:
            raise RuntimeError("Provided number of epochs does not mention number of epochs during initialization!")

        # Configure and Create Parameter paths
        if str(dataset_name).lower() not in self.name.lower():
            old_name = self.name
            self.name = self.name + f"_{str(dataset_name).upper()}"
            if old_name in self.param_path:
                self.param_path = self.param_path.replace(old_name, self.name)

        # Import either notebook or normal tqdm
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        # Prepare Models
        self.dis.train()
        self.gen.train()

        # Create Tensorboard connection
        writer = SummaryWriter(config.BASE_DIR + 'runs/' + datetime.now().strftime('%b-%d_%X_') + self.name
                               ) if tensorboard else NullWriter()

        pbar_epochs = tqdm(range(1, epochs + 1), desc="Epochs")
        d_steps, g_steps, batches_done = 0, 0, 1
        for epoch in pbar_epochs:
            with BatchMemoryManager(
                    data_loader=dataloader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=self.opt_g
            ) if self.dp else nullcontext(dataloader) as dataloader:
                pbar_batches = tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Batches")
                for i, data in pbar_batches:
                    batch_start = timer()

                    real, lengths, labels = split_data(data)

                    # Configure input
                    if isinstance(real, Tensor) and real.dim() == 3:
                        # Add feature dimension in case of single feature
                        real = [real, ]
                    real = [x.to(self.device) for x in real]

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    self.dis.train()
                    self.opt_d.zero_grad()

                    # Generate a batch of synthetic trajectories
                    # (For training of discriminator --> Generator in Eval mode)
                    self.gen.eval()  # Need to be set back to training mode later
                    noise = self.get_noise(real_trajs=real, traj_shaped_noise=self.traj_shaped_noise)
                    generated = [x.detach() for x in self.gen(noise)]

                    if self.wgan:
                        # (Improved) Wasserstein GAN
                        d_real = torch.mean(self.dis(real))
                        d_fake = torch.mean(self.dis(generated))
                        d_loss = -d_real + d_fake  # Vanilla WGAN loss
                        if self.gradient_penalty:
                            gradient_penalty = compute_gradient_penalty(
                                self.dis, real=real, synthetic=generated, lengths=lengths, lp=self.lipschitz_penalty)
                            d_loss += lambda_gp * gradient_penalty
                    else:
                        # Discriminator Ground Truth (real=0, fake=1)
                        batch_size = len(real[0])
                        real_labels = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
                        syn_labels = torch.ones((batch_size, 1), device=self.device, dtype=torch.float32)
                        real_loss = self.dis_loss(self.dis(real), real_labels)
                        fake_loss = self.dis_loss(self.dis(generated), syn_labels)
                        d_loss = (real_loss + fake_loss) / 2

                        # Add L1 regularizer for LSTM recurrent kernel as in TF model
                        if self.use_regularizer:
                            d_loss = d_loss + l1_regularizer(weights=self.dis.lstm.weight_hh_l0, l1=0.02)

                    d_loss.backward()
                    self.opt_d.step()
                    d_steps += 1

                    # Only if WGAN w/o gradient penalty used
                    if self.wgan and not self.gradient_penalty:
                        clip_discriminator_weights(dis=self.dis, clip_value=clip_value)

                    g_loss = None
                    if batches_done % n_critic == 0:
                        # -----------------
                        #  Train Generator
                        # -----------------
                        self.gen.train()  # Switch training mode on for generator
                        for _ in range(n_generator):
                            self.opt_g.zero_grad()

                            # Sample noise
                            noise = self.get_noise(real_trajs=real, traj_shaped_noise=self.traj_shaped_noise)

                            generated = self.gen(noise)

                            if self.wgan:
                                g_loss = -torch.mean(self.dis(generated))
                            else:
                                # Create proper label in case discriminator's labels are noisy
                                real_labels = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)
                                g_loss = self.gen_loss(
                                    y_true=real_labels,
                                    y_pred=self.dis(generated),
                                    real_trajs=real,
                                    gen_trajs=generated
                                )

                                # Add L1 regularizer for LSTM recurrent kernel as in TF model
                                if self.use_regularizer:
                                    g_loss = g_loss + l1_regularizer(weights=self.gen.lstm.weight_hh_l0, l1=0.02)

                            g_loss.backward()
                            self.opt_g.step()
                            g_steps += 1

                    # Generator loss
                    if g_loss is not None:
                        writer.add_scalar("Loss/Gen", g_loss.item(), global_step=batches_done)
                    # Discriminator loss
                    d_loss = d_loss.item()
                    if self.wgan:
                        if self.gradient_penalty:
                            # Remove gradient penalty and plot separately
                            d_loss -= lambda_gp * gradient_penalty
                            writer.add_scalar("Loss/GP", gradient_penalty, global_step=batches_done)
                        # For WGAN, one has to plot -D_loss not D_loss according to the authors.
                        d_loss = -d_loss
                    writer.add_scalar("Loss/Dis", d_loss, global_step=batches_done)
                    if self.gen.lstm.weight_hh_l0.grad is not None:
                        writer.add_scalar('Grad_Gen/LSTM_HH', self.gen.lstm.weight_hh_l0.grad.norm(),
                                          global_step=batches_done)
                        writer.add_scalar('Grad_Gen/LSTM_IH', self.gen.lstm.weight_ih_l0.grad.norm(),
                                          global_step=batches_done)
                    writer.add_scalar('Grad_Dis/LSTM_HH', self.dis.lstm.weight_hh_l0.grad.norm(),
                                      global_step=batches_done)
                    writer.add_scalar('Grad_Dis/LSTM_IH', self.dis.lstm.weight_ih_l0.grad.norm(),
                                      global_step=batches_done)
                    writer.add_scalar('Grad_Dis/OUT', self.dis.output_layer[0].weight.norm(),
                                      global_step=batches_done)

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
                                gen_samples=generated[0],
                                batches_done=batches_done,
                                notebook=notebook,
                                tensorboard=tensorboard,
                                writer=writer,
                            )
                        else:
                            # Trajectory Dataset
                            utils.visualize_trajectory_samples(
                                gen_samples=generated[0],  # Only latlon
                                real_samples=real[0],  # Only latlon
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

    def generate(self,
                 num: int,
                 max_length: int,
                 batch_size: int = MAX_PHYSICAL_BATCH_SIZE,
                 to_numpy: bool = True):
        """
        Generate a specified number of synthetic trajectories with a given maximum length.

        :param num: The total number of synthetic trajectories to generate.
        :param max_length: The maximum length (number of time steps) for each trajectory.
        :param batch_size: The size of each batch to be processed. Defaults to the maximum physical batch size.
                           Generation will use this batch size unless the remaining number of trajectories to
                           generate is smaller, in which case it will adjust for the last batch.
        :param to_numpy: If True, predictions are returned as numpy arrays; if False, as PyTorch tensors.
                         Defaults to True.

        :return: A list containing the generated synthetic trajectories for each feature. The type of the
                 contents is determined by the `to_numpy` parameter.
        """
        generated = [[] for _ in self.features]
        self.gen.eval()
        with torch.no_grad():
            for _ in range(math.ceil(num / batch_size)):
                noise = self.get_noise(
                    batch_size=min(batch_size, num - len(generated[0])),
                    num_time_steps=max_length,
                    feature_dims=[self.vocab_size[f] for f in self.features],
                    traj_shaped_noise=self.traj_shaped_noise)
                syn_trajs = [x.detach() for x in self.gen(noise)]
                if to_numpy:
                    syn_trajs = [x.cpu().numpy() for x in syn_trajs]
                for i, _ in enumerate(self.features):
                    generated[i].extend(syn_trajs[i])

        return generated
