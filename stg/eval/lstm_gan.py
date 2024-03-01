#!/usr/bin/env python3
"""Examine the influence of the discriminator's feedback on the training process."""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

import pandas as pd
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from stg import config
from stg.datasets import TrajectoryDataset, random_split, pad_feature_first, \
    DATASET_CLASSES, get_dataset
from stg.datasets.base_dataset import SPATIAL_FEATURE
from stg.eval.lstm_convergence import LT_EPOCHS, LT_BATCHES
from stg.eval.shared import determine_epochs, fix_seeds
from stg.metrics import compute_data_preservation
from stg.models.lstm_trajgan import LSTM_TrajGAN
from stg.models.traj_loss import TrajLoss
from stg.utils import logger

# General Parameters
N_FOLD = 5
OUTPUT_DIR = config.RESULT_DIR + 'lstm_gan/'

# LSTM-TrajGAN Parameters
# LT_EPOCHS = 250  # Computed below
# LT_MAX_LEN = 144  # Dataset Dependent -> defined below
LT_LATENT_DIM = 100
LT_BATCH_SIZE = 256
LT_LR = 0.001
LT_BETA = 0.5
LT_NAME = 'LSTM_TrajGAN_PT'

fix_seeds(42)

# Get logger
log = logging.getLogger()


def train_measure(opt: dict,
                  epochs: int,
                  loss: TrajLoss,
                  all_data: TrajectoryDataset,
                  train_dl: DataLoader,
                  test_data: TrajectoryDataset
                  ) -> dict:
    # Create GAN
    gan = LSTM_TrajGAN(
        reference_point=all_data.reference_point,
        features=all_data.features,
        scale_factor=all_data.scale_factor,
        gpu=opt['gpu'],
        learning_rate=LT_LR,
        beta=LT_BETA,
        model_name=LT_NAME + "_" + str(opt['dataset']).upper(),
    )

    # Change loss
    gan.gen_loss = loss

    # Train GANs
    if opt['load']:
        log.warning("Loading LSTM-TrajGAN parameters from file. Should be used for debugging only!")
        gan.load_parameters(epoch=epochs)
    else:
        gan.training_loop(
            train_dl,
            epochs=epochs,
            save_freq=0,  # Don't save
            print_training=False
        )

    # Predict GAN
    predicted_df = gan.predict_and_convert(test_data)

    # Create DataFrames for entire dataset
    test_df = pd.concat(test_data.originals.values())
    results = compute_data_preservation(
        test_df, predicted_df,
        categorical_features=[feature for feature in all_data.features[1:] if feature != SPATIAL_FEATURE])

    return results


def run_eval(opt: dict) -> None:
    start_time = timer()
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    dataset = opt['dataset']
    opt['dataset'] = opt['dataset'] if not opt['spatial'] else opt['dataset'] + "_spatial"
    # Output file uses current date and time - so we won't overwrite anything and can concatenate later
    OUTPUT_FILE = OUTPUT_DIR + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{opt['dataset']}.csv"
    META_FILE = OUTPUT_DIR + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{opt['dataset']}.json"
    print(f"Saving results to:\t", OUTPUT_FILE)

    # Load dataset
    all_data = get_dataset(dataset, latlon_only=opt['spatial'])

    # Print Dataset Information
    print("Total Samples:\t\t", len(all_data))
    print("Reference Point:\t", all_data.reference_point)
    print("Scaling Factor:\t\t", all_data.scale_factor)

    for run in range(1, opt['runs'] + 1):
        print("#" * 15, "Starting Run #", run, "#" * 15)
        # Randomly split into train and test
        train_data: TrajectoryDataset
        test_data: TrajectoryDataset
        train_data, test_data = random_split(all_data, [2 / 3, 1 / 3])

        print("LSTM-TrajGAN Train:\t", len(train_data))
        print("LSTM-TrajGAN Gen:\t", len(test_data))
        # Make sure we didn't lose any samples
        assert len(train_data) + len(test_data) == len(all_data), "Splitting failed."
        # Make sure every TID is only in one of the two datasets
        assert len(set(train_data.tids).intersection(set(test_data.tids))) == 0, "Splitting failed."

        # Create DataLoaders
        train_dl = DataLoader(train_data, batch_size=LT_BATCH_SIZE, shuffle=True, drop_last=True,
                              collate_fn=pad_feature_first)

        # Determine epochs
        epochs, num_batches = determine_epochs(opt['epochs'], opt['num_batches'], len(train_dl))

        p_cat = 1 if 'category' in all_data.features else 0
        p_hour = 1 if 'hour' in all_data.features else 0
        p_dow = 1 if 'day' in all_data.features else 0
        losses = {
            'STD': TrajLoss(p_cat=p_cat, p_hour=p_hour, p_dow=p_dow),
            'NO BCE': TrajLoss(p_bce=0, p_cat=p_cat, p_hour=p_hour, p_dow=p_dow),
            'BCE ONLY': TrajLoss(p_latlon=0, p_cat=0, p_hour=0, p_dow=0),
        }

        # Train GAN and measure data preservation
        results = {
            'run': [],
            'dataset': [],
            'loss': [],
            'epochs': [],
            'num_batches': [],
            'num_trajectories': [],
            'runtime': [],
        }
        for loss in losses:
            print("Training with Loss:\t", loss)
            start_time_run = timer()
            tmp_results = train_measure(opt, epochs, losses[loss], all_data, train_dl, test_data)

            # prepend loss name to all tmp_results
            for key in tmp_results:
                if key in results:
                    results[key].append(tmp_results[key])
                else:
                    results[key] = [tmp_results[key]]

            # Add run information
            results['run'].append(run)
            results['dataset'].append(opt['dataset'])
            results['loss'].append(loss)
            results['epochs'].append(epochs)
            results['num_batches'].append(num_batches)
            results['num_trajectories'].append(len(test_data))
            results['runtime'].append(timer() - start_time_run)

        # Save results
        pd.DataFrame(results).to_csv(OUTPUT_FILE, mode='a', header=not Path(OUTPUT_FILE).exists(), index=False)
        print("Wrote results to:\t", OUTPUT_FILE)

        # Create Meta Data
        metadata = {
            'dataset': opt['dataset'],
            'num_trajectories': len(test_data),
            'num_batches': num_batches,
            'epochs': epochs,
            'runtime_total': timer() - start_time,
            'runtime_avg': (timer() - start_time) / run,
            'losses': list(losses.keys()),
            'features': all_data.features,
            'reference_point': all_data.reference_point,
            'scale_factor': all_data.scale_factor,
        }
        # Save Metadata
        with open(META_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)

        print("Runtime:\t\t", timedelta(seconds=(timer() - start_time) / run))
    print("Total Runtime:\t\t", timedelta(seconds=timer() - start_time))


if __name__ == '__main__':
    logger.configure_root_loger(logging_level=logging.WARNING)  # Configure root logger
    # Argument parser via argparse
    parser = argparse.ArgumentParser(description='LSTM-TrajGAN Loss Evaluation.')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='GPU to use.')
    parser.add_argument('-r', '--runs', type=int, default=N_FOLD, help='Number of runs to perform.')
    parser.add_argument('--load', action='store_true', help='Load LSTM-TrajGAN parameters from file.')
    parser.add_argument('--print', type=str, default=None, metavar='FILE',
                        help='Print summary of the results from FILE.')
    parser.add_argument('-d', '--dataset', type=str, default='fs', choices=DATASET_CLASSES.keys(),
                        help='Dataset to use for evaluation.')
    parser.add_argument('-e', '--epochs', type=int, default=LT_EPOCHS, help='Number of epochs to train.')
    parser.add_argument('-b', '--num_batches', type=int, default=LT_BATCHES,
                        help='Number of batches to train for. Takes precedence over number of epochs.')
    parser.add_argument('--spatial', action='store_true', help='Use spatial features only.')
    opt = vars(parser.parse_args())
    if opt['print'] is not None:
        df = pd.read_csv(OUTPUT_DIR + opt['print'])
        print(df.groupby(['dataset', 'num_batches', 'loss']).mean())
    else:
        run_eval(opt)
