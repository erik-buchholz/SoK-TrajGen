#!/usr/bin/env python3
"""Implement distance measures used in the evaluation of the generated trajectories"""

import logging
from typing import List, Sequence

import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit

logging.getLogger()


def compute_list_differences(original: Sequence, generated: Sequence) -> int:
    """Compute the number of differences between two lists."""
    if not len(original) == len(generated):
        raise ValueError(f"Sequences must have the same length: {len(original)} != {len(generated)}")
    return sum([1 for o, g in zip(original, generated) if o != g])


def compute_data_preservation(original: pd.DataFrame,
                              generated: pd.DataFrame,
                              categorical_features: List[str],
                              print_results: bool = False,
                              lat_column: str = 'lat',
                              lon_column: str = 'lon'
                              ) -> dict:
    """Compute how close the generated data is to the original data."""
    if not len(original) == len(generated):
        raise ValueError(f"Sequences must have the same length: {len(original)} != {len(generated)}")
    result = {}
    # Haversine distance between locations
    haversine_distances = haversine_vector(original[[lat_column, lon_column]], generated[[lat_column, lon_column]],
                                           Unit.METERS)
    haversine_mean = haversine_distances.mean()
    result["haversine_mean [m]"] = haversine_mean

    # Euclidean distance between locations
    euclidean_distances = np.linalg.norm(
        original[[lat_column, lon_column]].to_numpy() - generated[[lat_column, lon_column]].to_numpy())
    euclidean_mean = euclidean_distances.mean()
    result["euclidean_mean"] = euclidean_mean

    # Determine the maximum width for alignment
    max_width = max(len("Total Points:"), len("Haversine Distance:"), len("Euclidean Distance:"),
                    *[len(feature) + 1 for feature in categorical_features])

    if print_results:
        # Print aligned results
        print(f"{'Total Points:':<{max_width}}\t{len(original)}")
        print(f"{'Haversine Distance:':<{max_width}}\t{haversine_mean:.2f} m")
        print(f"{'Euclidean Distance:':<{max_width}}\t{euclidean_mean:.2f}")

    if 'timestamp' in original.columns:
        # Create hour and day columns
        if 'hour' not in original.columns:
            original['hour'] = original['timestamp'].dt.hour
        if 'day' not in original.columns:
            original['day'] = original['timestamp'].dt.dayofweek

    # Categorical Features
    for feature in categorical_features:
        absolute_diff = compute_list_differences(original[feature], generated[feature])
        relative_diff = absolute_diff / len(original[feature])
        result[f"{feature}_abs"] = absolute_diff
        result[f"{feature}_rel"] = relative_diff
        if print_results:
            print(f"{feature + ':':<{max_width}}\t{absolute_diff:02} / {len(original):3} ({relative_diff * 100:.0f}%)")
    return result
