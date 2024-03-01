#!/usr/bin/env python3
"""What happens if LSTM-TrajGAN is trained for long periods of time?"""
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd
from torch.utils.data import DataLoader

from stg import config
from stg.datasets import TrajectoryDataset, pad_feature_first, random_split, \
    DATASET_CLASSES, get_dataset
from stg.datasets.base_dataset import SPATIAL_FEATURE
from stg.eval.shared import determine_epochs, fix_seeds
from stg.metrics import compute_data_preservation
from stg.models.lstm_trajgan import LSTM_TrajGAN
from stg.utils import logger

# General Parameters
N_FOLD = 5
OUTPUT_DIR = config.RESULT_DIR + 'lstm_conv/'

# LSTM-TrajGAN Parameters
LT_EPOCHS = 250  # Can be overwritten via command line
LT_BATCHES = 2000  # Can be overwritten via command line
# LT_MAX_LEN = 144  # Dataset Dependent -> defined below
LT_LATENT_DIM = 100
LT_BATCH_SIZE = 256
LT_LR = 0.001
LT_BETA = 0.5
LT_NAME = 'LSTM_TrajGAN_PT'

fix_seeds(42)

# Get logger
log = logging.getLogger()


def run_eval(opt: dict) -> None:
    start_time = timer()
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Output file uses current date and time - so we won't overwrite anything and can concatenate later
    dataset = opt['dataset']
    opt['dataset'] = opt['dataset'] if not opt['spatial'] else opt['dataset'] + "_spatial"
    OUTPUT_FILE = OUTPUT_DIR + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{opt['dataset']}_{opt['num_batches']}.csv"
    META_FILE = OUTPUT_DIR + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{opt['dataset']}_{opt['num_batches']}.json"
    print("Writing results to:\t", OUTPUT_FILE)


    # Load dataset
    all_data = get_dataset(dataset, latlon_only=opt['spatial'])

    # Print Dataset Information
    print("Total Samples:\t\t", len(all_data))
    print("Reference Point:\t", all_data.reference_point)
    print("Scaling Factor:\t\t", all_data.scale_factor)

    for run in range(1, opt['runs'] + 1):
        print("#" * 15, "Starting Run #", run, "#" * 15)
        start_time_run = timer()

        # Randomly split into train and test
        train_data: TrajectoryDataset
        test_data: TrajectoryDataset
        train_data, test_data = random_split(all_data, [2 / 3, 1 / 3])

        print("Training Set:\t\t", len(train_data))
        print("Test Set:\t\t", len(test_data))
        assert len(train_data) + len(test_data) == len(all_data), "Splitting failed."
        # Make sure every TID is only in one of the two datasets
        assert len(set(train_data.tids).intersection(set(test_data.tids))) == 0, "Splitting failed."

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

        # Create DataLoaders
        train_dl = DataLoader(train_data, batch_size=LT_BATCH_SIZE, shuffle=True, drop_last=True,
                              collate_fn=pad_feature_first)

        # Determine epochs
        epochs, num_batches = determine_epochs(opt['epochs'], opt['num_batches'], len(train_dl))

        # Train GAN
        if opt['load']:
            log.warning("Loading LSTM-TrajGAN parameters from file. Should be used for debugging only!")
            gan.load_parameters(epoch=epochs)
        else:
            gan.training_loop(
                train_dl,
                epochs=epochs,
                save_freq=epochs,  # Save only the last model state
                print_training=False
            )

        # Predict GAN
        predicted_df = gan.predict_and_convert(test_data)
        # predicted = helpers.df2trajectory_dict(predicted_df, 'tid')

        # Create DataFrames for entire dataset
        test_df = pd.concat(test_data.originals.values())
        results = compute_data_preservation(test_df, predicted_df,
                                            categorical_features=[feature for feature in all_data.features[1:] if
                                                                  feature != SPATIAL_FEATURE],
                                            print_results=True)
        results.update(
            {'run': run,
             'dataset': opt['dataset'],
             'epochs': epochs,
             'num_batches': num_batches,
             'num_trajectories': len(test_data),
             'num_points': len(test_df),
             'runtime': timer() - start_time_run}
        )
        for key in results:
            results[key] = [results[key]]

        # Generate metadata
        metadata = {
            'dataset': opt['dataset'],
            'epochs': epochs,
            'num_batches': num_batches,
            'n_train': len(train_data),
            'n_test': len(test_data),
            'features': all_data.features,
            'latlon_only': all_data.latlon_only,
            'reference_point': all_data.reference_point,
            'scale_factor': all_data.scale_factor,
            'runtime total': timer() - start_time,
            # Compute average runtime per run
            'runtime run': (timer() - start_time_run) / run,
        }


        # Save results
        pd.DataFrame(results).to_csv(OUTPUT_FILE, mode='a', header=not Path(OUTPUT_FILE).exists(), index=False)
        # Save Metadata
        with open(META_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)

        print("Wrote results to:\t", OUTPUT_FILE)
        # Print Run Runtime
        print(f"Runtime:\t\t{timedelta(seconds=int(timer() - start_time_run))}")

    print(f"Total Time:\t\t{timedelta(seconds=int(timer() - start_time))}")



if __name__ == '__main__':
    logger.configure_root_loger(logging_level=logging.WARNING)  # Configure root logger
    # Argument parser via argparse
    parser = argparse.ArgumentParser(description='LSTM-TrajGAN Convergence.')
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
        # Print Average of all runs
        print(df.groupby(['dataset', 'epochs', 'num_batches']).mean())
    else:
        run_eval(opt)
