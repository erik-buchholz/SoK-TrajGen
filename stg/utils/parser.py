#!/usr/bin/env python3
""" """
import json
import logging

from stg import config

log = logging.getLogger()
IGNORE_KEYS = ['gpu', 'summary', 'save']  # Run-specific values


def load_config_file(opt: dict) -> dict:
    """
    Write config from JSON file and overwrite the provided arguments.
    Inspired by:
    https://github.com/ratschlab/RGAN/blob/master/utils.py
    :param opt: Dict containing configuration values, MUST contain key 'config' defining the configuration name
    :return: Dict containing updated parameters
    """
    config_path = config.BASE_DIR + "config/" + opt['config']
    # Append .json if not present
    if not config_path.endswith('.json'):
        config_path += '.json'
    log.info(f"Loading configuration from '{config_path}'")
    settings_loaded = json.load(open(config_path, 'r'))
    # check for settings missing in file
    for key in opt.keys():
        if key not in settings_loaded and key not in IGNORE_KEYS:
            log.warning(f"Key '{key}' not found in config file - adopting value from command line defaults: {opt[key]}")
            # overwrite parsed/default settings with those read from file, allowing for
            # (potentially new) default settings not present in file
    opt.update(settings_loaded)
    return opt


def write_config_file(opt: dict) -> None:
    """

    :param opt: Dict containing configuration values, MUST contain key 'config' defining the configuration name
    :return: None
    """
    config_path = config.BASE_DIR + "config/" + opt['config'] + '.json'
    # Remove session keys
    for key in IGNORE_KEYS:
        if key in opt:
            del opt[key]
    json.dump(opt, open(config_path, 'w'), indent=4)
    log.info(f"Wrote configuration to '{config_path}.")
