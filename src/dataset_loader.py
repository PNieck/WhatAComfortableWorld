"""Dataset loading utilities for floor plan data.

This module provides functions to load floor plan datasets from various sources
(text files, MAT files) and manage train/validation/test splits.
"""

import scipy.io as sio

from datasets import load_dataset, DatasetDict

from enum import IntFlag, auto


INVALID_FLOOR_PLANS_FILE = "data/invalid_floor_plans.txt"
IMPERFECT_FLOOR_PLANS_FILE = "data/imperfect_floor_plans.txt"


class Split(IntFlag):
    """
    Enumeration of dataset splits.

    Uses IntFlag to allow combining splits with bitwise OR operations.
    """
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def load_floor_plans_dataset(path: str, splits: Split = Split.TRAIN | Split.VALID | Split.TEST) -> DatasetDict:
    """
    Load floor plan sequences from text files.

    Loads one or more dataset splits (train, validation, test) from a directory
    containing corresponding .txt files. Each file contains one floor plan sequence
    per line.

    :param path: Path to the directory containing split files
    :param splits: Bitmask specifying which splits to load (default: all)
    :return: DatasetDict with loaded datasets keyed by split name
    """
    data_files = {}

    if splits & Split.TRAIN:
        data_files["train"] = path + "/train.txt"
    
    if splits & Split.TEST:
        data_files["test"] = path + "/test.txt"

    if splits & Split.VALID:
        data_files["valid"] = path + "/validation.txt"
    
    dataset = load_dataset(
        "text",
        data_files=data_files
    )

    return dataset


def _load_floor_plan_names(path: str) -> set[str]:
    """
    Load floor plan names from a text file.

    Reads a file where each line is a floor plan name, ignoring comments
    (lines starting with '#') and empty lines. Strips whitespace from names.

    :param path: Path to the text file
    :return: Set of floor plan names; empty set if file not found
    """
    try:
        with open(path, mode='r') as file:
            plans = file.readlines()
    except FileNotFoundError:
        print(f"WARNING: cannot load floor plans from {path}")
        return set()

    # Remove all white spaces from the end 
    plans = [line.rstrip() for line in plans]

    # Remove comments and empty strings 
    plans = [plan for plan in plans if plan != "" and not plan.startswith("#")]

    return set(plans)


def load_invalid_floor_plans_names() -> set[str]:
    """
    Load names of invalid floor plans.

    Loads the configured list of invalid floor plan names from the standard
    invalid floor plans file.

    :return: Set of invalid floor plan names
    """
    return _load_floor_plan_names(INVALID_FLOOR_PLANS_FILE)


def load_imperfect_floor_plans_names() -> set[str]:
    """
    Load names of imperfect floor plans.

    Loads the configured list of imperfect floor plan names from the standard
    imperfect floor plans file.

    :return: Set of imperfect floor plan names
    """
    return _load_floor_plan_names(IMPERFECT_FLOOR_PLANS_FILE)


def load_dataset_from_mat_file(path: str, exclude_invalid=True, exclude_imperfect=False):
    """
    Load floor plan dataset from a MATLAB .mat file.

    Loads floor plan data from a MAT file and optionally filters out invalid or
    imperfect plans based on configured exclusion lists.

    Information about output format can be found there:
    https://github.com/HanHan55/Graph2plan/tree/master/DataPreparation

    :param path: Path to the .mat file
    :param exclude_invalid: If True, exclude plans marked as invalid (default: True)
    :param exclude_imperfect: If True, exclude plans marked as imperfect (default: False)
    :return: List of floor plan objects in mat file format, optionally filtered
    """
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)['data']

    if exclude_invalid:
        invalid_plans = load_invalid_floor_plans_names()
        data = [plan for plan in data if not plan.name in invalid_plans]
    
    if exclude_imperfect:
        imperfect_plans = load_imperfect_floor_plans_names()
        data = [plan for plan in data if not plan.name in imperfect_plans]

    return data
