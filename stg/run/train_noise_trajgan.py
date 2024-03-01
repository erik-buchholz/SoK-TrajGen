#!/usr/bin/env python3
"""Train RGAN Model."""
import json

from torch.utils.data import DataLoader

from stg.datasets import mnist_sequential, DatasetModes
from stg.datasets.dataset_factory import Datasets, get_dataset
from stg.datasets.padding import pad_feature_first
from stg.models import Noise_TrajGAN
from stg import parser
from stg.models.lstm_trajgan import VOCAB_SIZE, EMBEDDING_SIZE
from stg.utils.parser import load_config_file
from stg.utils import logger

if __name__ == '__main__':
    opt = vars(parser.get_trajgan_parser().parse_args())
    if 'config' in opt and opt['config'] is not None:
        opt = load_config_file(opt)
    print("Arguments: ", json.dumps(opt, indent=4))

    log = logger.configure_root_loger(logging_level=opt.pop('logging_lvl'))

    # Prepare Data
    collate_fn = None
    dataset_name = opt.pop('dataset')
    if dataset_name == Datasets.MNIST_SEQUENTIAL:
        dataset = mnist_sequential(28)
        vocab_size = {'mnist': 28}
        embedding_size = {'mnist': 28}
    else:
        # Trajectory Dataset
        dataset = get_dataset(
            dataset_name=dataset_name,
            mode=DatasetModes.ALL,
            latlon_only=False,
            normalize=True,
            return_labels=False,
            keep_original=False
        )
        # Padding
        collate_fn = pad_feature_first
        # Vocabulary / Embedding size required for MNIST Sequential
        vocab_size = VOCAB_SIZE
        embedding_size = EMBEDDING_SIZE

    # Create Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=opt.pop('batch_size'),
                            drop_last=True,
                            shuffle=opt.pop('shuffle'),
                            collate_fn=collate_fn,
                            pin_memory=True
                            )

    # Create GAN
    model = Noise_TrajGAN(
        # General Options
        gpu=opt.pop('gpu'),
        name=opt.pop('name'),
        # Data Options
        features=dataset.features,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        # Architecture Options
        recurrent_layers=opt.pop('n_layers'),
        latent_dim=opt.pop('latent_dim'),
        noise_dim=opt.pop('noise_dim'),
        traj_shaped_noise=opt.pop('traj_shaped_noise'),
        # Optimizer Options
        opt_d=opt.pop('opt_d'),
        opt_g=opt.pop('opt_g'),
        lr_d=opt.pop('lr_d'),
        lr_g=opt.pop('lr_g'),
        beta1=opt.pop('beta1'),
        beta2=opt.pop('beta2'),
        # GAN Loss options
        wgan=opt.pop('wgan'),
        gradient_penalty=opt.pop('gradient_penalty'),
        lipschitz_penalty=opt.pop('lp'),
        # Privacy Options
        dp=opt.pop('dp'),
        epsilon=opt.pop('epsilon'),
        delta=opt.pop('delta'),

    )

    # Remove the admin parameters from the dictionary
    # Note that we used pop() to remove the parameters from the dictionary above
    opt = parser.clean_args(opt)

    # Training
    model.training_loop(dataloader=dataloader,
                        dataset_name=dataset_name,
                        notebook=False,
                        **opt)  # By using **opt, we make sure that all arguments were used
