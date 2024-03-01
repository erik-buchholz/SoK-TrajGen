#!/usr/bin/env python3
"""Implements LSTM-TrajGAN with PyTorch instead of TensorFlow"""
import argparse
import logging
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from timeit import default_timer
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader

from stg import config
from stg.datasets.base_dataset import TrajectoryDataset, DatasetModes, SPATIAL_COLUMNS
from stg.datasets.fs_nyc import FSNYCDataset, PATH_ALL
from stg.datasets.padding import pad_feature_first
from stg.models.trajGAN import TrajGAN
from stg.models.traj_loss import get_trajLoss
from stg.models.utils import l1_regularizer
from stg.utils import logger, helpers
from stg.utils.helpers import count_parameters_torch

# CONSTANTS ####################################################################
LSTM_TrajGAN_FEATURES = ['latlon', 'hour', 'day', 'category']
VOCAB_SIZE = {
    'latlon': 2,
    'hour': 24,
    'day': 7,
    'category': 10
}
EMBEDDING_SIZE = {
    'latlon': 64,
    'hour': 24,
    'day': 7,
    'category': 10
}
log = logging.getLogger()

# Constants from Paper
# MAX_LEN = 144  # Only valid for Foursquare NYC Dataset
LATENT_DIM = 100  # Noise vector length not actually used for model's latent dimension
EPOCHS = 250  # Equivalent to 2000 batches in case of Foursquare NYC Dataset
LEARNING_RATE = 0.001
BETA = 0.5
BATCH_SIZE = 256
MAX_PHYSICAL_BATCH_SIZE = 3000


class Generator(nn.Module):
    def __init__(
            self,
            features: List[str],
            vocab_size: Dict[str, int],
            embedding_size: Dict[str, int],
            noise_len: int,
            scale_factor: (float, float),
            device: str,
            dp: bool = False,
            scale_in_model: bool = False
    ):
        super().__init__()

        # Store (hyper)parameters
        self.features = features
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.noise_len = noise_len
        self.scale_factor = scale_factor
        self.scale_in_model = scale_in_model

        # Create Model
        self.embedding_layers = nn.ModuleDict()
        for i, feature in enumerate(self.features):
            self.embedding_layers[feature] = nn.Sequential(
                nn.Linear(vocab_size[feature], embedding_size[feature], bias=True, dtype=torch.float32),
                nn.ReLU()
            )
        feature_len = sum(self.embedding_size[f] for f in self.features) + self.noise_len
        self.feature_fusion = nn.Sequential(nn.Linear(feature_len, 100, bias=True, dtype=torch.float32), nn.ReLU())
        if dp:
            self.lstm = DPLSTM(100, 100, batch_first=True)
        else:
            self.lstm = nn.LSTM(100, 100, batch_first=True, dtype=torch.float32)
        output_latlon = nn.Sequential(
            nn.Linear(100, 2, bias=True, dtype=torch.float32),
            nn.Tanh()
        )
        self.output_layers = nn.ModuleDict({
            'latlon': output_latlon,
        })
        for feature in self.features[1:]:
            self.output_layers[feature] = nn.Sequential(
                nn.Linear(100, self.vocab_size[feature], bias=True, dtype=torch.float32),
                nn.Softmax(dim=-1)
            )
        self.sf = torch.tensor(self.scale_factor, device=device)

    def forward(self, x):
        """

        :param x: List[Tensor] w/ shape (# features, batch_size, sequence_len, feature_size)
        :return: outputs.shape = (num_features, batch_size, time_steps, feature_size)
        """
        # Embedding Layer
        embeddings = []
        for i, feature in enumerate(self.features):
            embeddings.append(self.embedding_layers[feature](x[i].to(dtype=torch.float32)))
        # Add noise, too
        noise = x[-1].reshape(len(x[-1]), 1, -1)
        noise = noise.expand(-1, x[0].shape[-2], -1)
        embeddings.append(noise)
        # Feature Fusion
        concat = torch.cat(embeddings, dim=-1)
        fusion = self.feature_fusion(concat)
        # LSTM Layer
        lstm, _ = self.lstm(fusion)
        # Output Layer
        if self.scale_in_model:
            latlon = self.output_layers['latlon'](lstm) * self.sf
        else:
            latlon = self.output_layers['latlon'](lstm)
        outputs = [latlon, ]
        for feature in self.features[1:]:
            outputs.append(self.output_layers[feature](lstm))

        return outputs


class Discriminator(nn.Module):
    def __init__(
            self,
            features: List[str],
            vocab_size: Dict[str, int],
            embedding_size: Dict[str, int],
            dp: bool = False
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
        self.feature_fusion = nn.Sequential(nn.Linear(feature_len, 100, dtype=torch.float32), nn.ReLU())
        if dp:
            self.lstm = DPLSTM(100, 100, batch_first=True)
        else:
            self.lstm = nn.LSTM(100, 100, batch_first=True, dtype=torch.float32)
        self.output_layer = nn.Sequential(
            nn.Linear(100, 1, bias=True, dtype=torch.float32),
            nn.Sigmoid()
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


class LSTM_TrajGAN(TrajGAN):
    def __init__(
            self,
            reference_point: (float, float),
            scale_factor: (float, float),
            features: List[str] = LSTM_TrajGAN_FEATURES,
            vocab_size: Dict[str, int] = VOCAB_SIZE,
            embedding_size: Dict[str, int] = EMBEDDING_SIZE,
            latent_dim: int = 100,
            param_path: str = None,
            model_name: str = 'LSTM_TrajGAN_PT',
            learning_rate: float = LEARNING_RATE,
            beta: float = BETA,
            use_regularizer: bool = True,
            gpu: int = None,
            dp: bool = False,
            privacy_accountant: str = "rdp",
            scale_in_model: bool = False,
            lr_scheduler: bool = False
    ):
        super().__init__()

        # Store Data
        self.reference_point = reference_point
        self.scale_factor = scale_factor
        self.features = features
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.latent_dim = latent_dim
        self.use_regularizer = use_regularizer
        self.dp = dp
        self.scale_in_model = scale_in_model
        self.lr_scheduler = lr_scheduler

        if gpu is not None and gpu > -1 and torch.cuda.is_available():
            device = f"cuda:{gpu}"
        else:
            device = 'cpu'
        # Create components
        self.gen = Generator(
            features=features,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            noise_len=latent_dim,
            scale_factor=self.scale_factor,
            device=device,
            dp=self.dp,
            scale_in_model=self.scale_in_model
        ).to(device)
        self.dis = Discriminator(
            features=features,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            dp=self.dp
        ).to(device)

        # Loss functions
        self.dis_loss = torch.nn.BCELoss()
        self.gen_loss = get_trajLoss(features=self.features, weight=1)

        # Optimizers
        self.opt_d = torch.optim.Adam(self.dis.parameters(), lr=learning_rate, betas=(beta, 0.999))
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(beta, 0.999))

        # Add learning rate scheduler after 1000 epochs
        if self.lr_scheduler:
            self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.opt_d, milestones=[1000], gamma=0.1)
            self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.opt_g, milestones=[1000], gamma=0.1)

        # Configure and Create Parameter paths
        if self.dp and 'dp' not in model_name.lower():
            # Add DP to name
            model_name = "DP_" + model_name
        self.name = model_name

        # Configure and Create Parameter paths
        self.prepare_param_paths(name=self.name, param_path=param_path)

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

    def forward(self, x):
        return self.gen(x)

    def check_dataset_normalization(self, dataset: TrajectoryDataset):
        """Verify if the dataset and the model use the same reference point and scaling factor, and raise a
        warning if not."""
        warning_raised = False
        if not np.allclose(dataset.reference_point, self.reference_point, atol=1e-5):
            warnings.warn(
                f"Dataset reference point ({dataset.reference_point}) does not match model reference point "
                f"({self.reference_point})."
            )
            warning_raised = True
        if not np.allclose(dataset.scale_factor, self.scale_factor, atol=1e-5):
            warnings.warn(
                f"Dataset scale factor ({dataset.scale_factor}) does not match model scale factor "
                f"({self.scale_factor})."
            )
            warning_raised = True
        if warning_raised:
            # Wait for user input
            input("Unequal reference point/scaling factor. Press Enter to continue...")

    def training_loop(self,
                      dataloader: DataLoader,
                      epochs: int,
                      save_freq: int = 10,
                      print_training: bool = True,
                      notebook: bool = False
                      ) -> None:
        """

        :param dataloader: DataLoader for training
        :param epochs: Number of epochs to train
        :param save_freq: After how many epochs to save. -1 to deactivate.
        :param print_training: Whether to print training progress
        :param notebook: Whether to use tqdm_notebook instead of tqdm
        :return:
        """
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        # Check if dataset normalization matches model normalization
        self.check_dataset_normalization(dataloader.dataset)

        # Ignore the two backward hook warnings as Opacus confirmed that these
        # can safely be ignored
        warnings.filterwarnings("ignore", message=".* non-full backward hook")

        # Fix issue with dataloader for large number of epochs
        sys.setrecursionlimit(10000)

        self.dis.train()
        self.gen.train()

        if self.dp and not self.dp_initialised:
            raise RuntimeError("Call .dp_init() before training!")
        if self.dp and self.epochs != epochs:
            raise RuntimeError("Provided number of epochs does not mentioned number of epochs during initialization!")

        pbar = tqdm(range(1, epochs + 1), desc="Training")
        for epoch in pbar:
            with BatchMemoryManager(
                    data_loader=dataloader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=self.opt_g
            ) if self.dp else nullcontext(dataloader) as dataloader:
                for i, real_trajs in enumerate(dataloader):
                    """
                    real_traj[i][j]
                    i € range(len(self.features): The feature
                    j € range(batch_size): The trajectory
                    """
                    start = default_timer()

                    assert len(real_trajs) == len(self.features), \
                        f"Got: {len(real_trajs)} features, but expected {len(self.features)}."
                    batch_size = len(real_trajs[0])

                    # Discriminator Ground Truth
                    real_labels = torch.ones((batch_size, 1), device=self.device, dtype=torch.float32)
                    syn_labels = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

                    # Configure input (Append noise)
                    real_trajs = [x.to(self.device) for x in real_trajs]
                    noise = torch.randn(size=(batch_size, self.latent_dim), device=self.device)
                    real_trajs.append(noise)

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    self.dis.train()
                    self.opt_d.zero_grad()

                    # Generate a batch of synthetic trajectories
                    # (For training of discriminator --> Generator in Eval mode)
                    log.debug(f"Generate {batch_size} synthetic trajectories...")
                    self.gen.eval()  # Need to be set back to training mode later
                    gen_trajs_bc = [x.detach() for x in self.gen(real_trajs)]
                    log.debug("...generation done.")

                    log.debug(f"Train discriminator on real samples...")
                    real_loss = self.dis_loss(self.dis(real_trajs[:len(self.features)]), real_labels)
                    # Slicing in order to cut off the noise again
                    log.debug(f"Train discriminator on generated samples...")
                    fake_loss = self.dis_loss(self.dis(gen_trajs_bc), syn_labels)
                    d_loss = (real_loss + fake_loss) / 2

                    # Add L1 regularizer for LSTM recurrent kernel as in TF model
                    if self.use_regularizer:
                        d_loss = d_loss + l1_regularizer(weights=self.dis.lstm.weight_hh_l0, l1=0.02)

                    d_loss.backward()
                    self.opt_d.step()

                    # -----------------
                    #  Train Generator
                    # -----------------
                    self.gen.train()  # Switch training mode on for generator
                    self.opt_g.zero_grad()

                    # Sample noise
                    noise = torch.randn(size=(batch_size, self.latent_dim), device=self.device)
                    real_trajs[-1] = noise  # Replace the noise to train the generator with fresh noise

                    # Generate a batch of images
                    gen_trajs = self.gen(real_trajs)

                    # Loss measures generator's ability to fool the discriminator
                    # g_loss = torch.nn.BCELoss(reduction='mean')(self.dis(gen_trajs), real_labels)
                    g_loss = self.gen_loss(
                        y_true=real_labels,
                        y_pred=self.dis(gen_trajs),
                        real_trajs=real_trajs[:len(self.features)],  # Without noise,
                        gen_trajs=gen_trajs,
                        pbar=pbar,
                        verbose=False
                    )
                    # Add L1 regularizer for LSTM recurrent kernel as in TF model
                    if self.use_regularizer:
                        g_loss = g_loss + l1_regularizer(weights=self.gen.lstm.weight_hh_l0, l1=0.02)

                    g_loss.backward()
                    self.opt_g.step()

                    msg = (
                        f"[Epoch {epoch:03d}/{epochs}] [Batch {i + 1:03d}/{len(dataloader)}] "
                        f"[D loss: {d_loss.item():.5f}] [G loss: {g_loss.item():.5f}] "
                    )
                    if self.dp:
                        msg += f"[Eps: {self.get_epsilon():.2f}] [Delta: {self.get_delta():.5f}]"
                    msg += f"in {default_timer() - start:.2f}s"
                    if print_training:  # Output on each batch
                        pbar.write(msg)

            if save_freq > 0 and epoch % save_freq == 0:
                self.save_parameters(epoch)

            if self.lr_scheduler:
                # Update learning rate scheduler at end of epoch (not batch!)
                self.scheduler_d.step()
                self.scheduler_g.step()

    def predict(self, dataloader: DataLoader, to_numpy: bool = True):
        """

        :param dataloader: test dataloader
        :param to_numpy: Predictions are returned as numpy arrays, otherwise as tensors
        :return:
        """
        # Check if dataset normalization matches model normalization
        self.check_dataset_normalization(dataloader.dataset)
        prediction = [[] for _ in self.features]
        self.gen.eval()  # Layers work in eval mode
        with torch.no_grad():  # No gradient computation --> speedup
            for real_trajs in dataloader:
                batch_size = len(real_trajs[0])
                real_trajs = [x.to(self.device) for x in real_trajs]
                noise = torch.randn(size=(batch_size, self.latent_dim), device=self.device)
                real_trajs.append(noise)
                syn_trajs = [x.detach() for x in self.gen(real_trajs)]
                if to_numpy:
                    syn_trajs = [x.cpu().numpy() for x in syn_trajs]
                for i, _ in enumerate(self.features):
                    prediction[i].extend(syn_trajs[i])

        return prediction

    def evaluate(self, test_dataset: TrajectoryDataset, batch_size: int = MAX_PHYSICAL_BATCH_SIZE) -> float:
        """Measure the performance of the model on a test set"""
        # Check if dataset normalization matches model normalization
        self.check_dataset_normalization(test_dataset)
        test_dl = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=pad_feature_first)
        losses = []
        self.gen.eval()  # Layers work in eval mode
        with torch.no_grad():  # No gradient computation --> speedup
            for real_trajs in test_dl:
                batch_size = len(real_trajs[0])
                real_labels = torch.ones((batch_size, 1), device=self.device, dtype=torch.float32)
                real_trajs = [x.to(self.device) for x in real_trajs]
                noise = torch.randn(size=(batch_size, self.latent_dim), device=self.device)
                real_trajs.append(noise)
                syn_trajs = self.gen(real_trajs)
                loss = self.gen_loss(
                    y_true=real_labels,
                    y_pred=self.dis(syn_trajs),
                    real_trajs=real_trajs[:len(self.features)],  # Without noise,
                    gen_trajs=syn_trajs
                )
                losses.append(loss)
        total_loss = sum(losses)
        # Note: We compute the loss over the entire test set, hence the result is only comparable for the same
        # set
        # noinspection PyUnresolvedReferences
        return float(total_loss.cpu().numpy())

    def predict_and_convert(self,
                            test_dataset: TrajectoryDataset,
                            batch_size: int = 512,
                            tid_label: str = 'tid',
                            uid_label: str = 'uid',
                            ) -> pd.DataFrame:
        # Check if dataset normalization matches model normalization
        self.check_dataset_normalization(test_dataset)
        # Shuffle is false to maintain the order
        test_dl = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             collate_fn=pad_feature_first, drop_last=False)

        prediction = self.predict(test_dl)

        # First index of predictions are the features, while
        # first index of dataset is the trajectory index
        assert len(prediction[0]) == len(test_dataset), "Prediction and dataset must have the same length."

        result = {
            tid_label: [],
            uid_label: [],
        }

        spatial_cols = test_dataset.columns
        latlon = []

        for n_feature, feature in enumerate(self.features):
            if feature != 'latlon':
                result[feature] = []
            for n_traj, trajectory in enumerate(prediction[n_feature]):
                # Remove the padding (pre-padding was used)

                # The trajectories in the test dataset are either a list with format
                # (n_features, n_timesteps, feature_dim), or a tensor of format (n_timesteps, feature_dim).
                original_trajectory = test_dataset[n_traj]
                if isinstance(original_trajectory, torch.Tensor) and len(original_trajectory.shape) == 2:
                    original_length = original_trajectory.shape[0]
                else:
                    original_length = len(original_trajectory[0])

                unpadded = trajectory[-original_length:]
                assert len(unpadded) == original_length, "Padding was not removed correctly."

                if n_feature == 0:
                    # Only do this once!
                    # Determine tid and uid
                    tid = test_dataset.tids[n_traj]
                    uid = test_dataset.uids[tid]
                    tid_array = np.full(original_length, tid)
                    uid_array = np.full(original_length, uid)
                    result[tid_label].extend(tid_array)
                    result[uid_label].extend(uid_array)

                if feature == 'latlon':
                    latlon.append(unpadded)
                else:
                    # We assume all other features are softmax outputs
                    result[feature].extend(np.argmax(unpadded, axis=1))

        # Concatenate latlon
        latlon = np.concatenate(latlon, axis=0)

        # Denormalize coordinates
        from stg.utils.data import denormalize_points
        latlon = denormalize_points(
            latlon,
            ref=test_dataset.reference_point,
            scale=test_dataset.scale_factor
        )
        result[spatial_cols[0]] = latlon[:, 0]
        result[spatial_cols[1]] = latlon[:, 1]

        assert len(result[tid_label]) == len(result[spatial_cols[0]]), \
            f"Length of tid and lat must be equal. TID: {len(result[tid_label])}, LAT: {len(result[spatial_cols[0]])}"
        assert len(result[uid_label]) == len(result[spatial_cols[0]]), \
            f"Length of uid and lat must be equal. UID: {len(result[uid_label])}, LAT: {len(result[spatial_cols[0]])}"

        # Create DataFrame
        df = pd.DataFrame(result)

        return df


if __name__ == '__main__':
    logger.configure_root_loger(logging.INFO, file=config.LOG_DIR + 'lstm_trajGAN_pytorch.log')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs of training")
    parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE, help="size of the batches")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--beta", type=float, default=BETA, help="Optimizer beta")
    parser.add_argument('--name', type=str, default='LSTM_TrajGAN_PT',
                        help="Model Name (determines parameter path)")
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM,
                        help="dimensionality of the noise space")
    parser.add_argument("--dp", action='store_true')
    parser.add_argument("--summary", action='store_true')
    parser.add_argument("-g", "--gpu", type=int, required=True, help="GPU ID to use")
    parser.add_argument("--no_reg", action='store_true', help="Deactivate L1 regularizer for LSTM")
    parser.add_argument("-s", "--save_freq", type=int, default=50, help="Save frequency")
    parser.add_argument("-p", "--silent", action='store_true', help="Show progress bar only.")
    parser.add_argument("--lr_scheduler", action='store_true', help="Use learning rate scheduler")
    opt = parser.parse_args()
    log.info(f"CMD line arguments: {opt}")

    if opt.summary:
        gan = LSTM_TrajGAN((0, 0), (1, 1), gpu=None)
        print("#" * 80, "\n", "Total Parameters of Generator: ", count_parameters_torch(gan.gen))
        print("#" * 80)
        print("#" * 80, "\n", "Total Parameters of Discriminator: ", count_parameters_torch(gan.dis))
        print("#" * 80)
    else:
        # Get reference point and scale factor from all data
        # Refer https://github.com/GeoDS/LSTM-TrajGAN/blob/master/train.py
        # This is technical a privacy leak. Using dataset.ref_point would be preferable,
        # however, does not work if the test set spans a larger area.
        # Best Case: Use a dataset-independent reference point
        columns = SPATIAL_COLUMNS
        _all = pd.read_csv(PATH_ALL)
        ref = helpers.get_ref_point(_all[columns])
        sf = helpers.get_scaling_factor(_all[columns], ref)
        # Make dict
        ref = {k: v for k, v in zip(columns, ref)}
        sf = {k: v for k, v in zip(columns, sf)}
        log.info(f"Reference point:\t\t{ref}")
        log.info(f"Scale factor:\t\t{sf}")

        dataset = FSNYCDataset(
            mode=DatasetModes.TRAIN,
            latlon_only=False,
            normalize=True,
            return_labels=False,
            reference_point=ref,
            scale_factor=sf
        )

        dl = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=pad_feature_first
        )
        gan = LSTM_TrajGAN(
            reference_point=[ref[c] for c in columns],
            scale_factor=[sf[c] for c in columns],
            gpu=opt.gpu,
            learning_rate=opt.lr,
            beta=opt.beta,
            model_name=opt.name,
            latent_dim=opt.latent_dim,
            use_regularizer=(not opt.no_reg),
            lr_scheduler=opt.lr_scheduler,
        )
        log.info("Start Training...")
        gan.training_loop(dataloader=dl, epochs=opt.epochs, save_freq=opt.save_freq, print_training=(not opt.silent))
        log.info("...Training done.")
