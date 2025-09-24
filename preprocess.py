import scipy.io as sio
import numpy as np

import sys
import argparse
import os
import yaml
import random
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src.mat_file import from_mat_file 
from src.sequence import to_sequence
from src.data_augmentation import permutate, rotation, symmetry, SymmetryType, RotationAngle


INVALID_FLOOR_PLANS_FILE = "data/invalid_floor_plans.txt"


def load_invalid() -> set[str]:
    try:
        with open(INVALID_FLOOR_PLANS_FILE, mode='r') as file:
            invalid_plans = file.readlines()
    except FileNotFoundError:
        print("WARNING: cannot load invalid floor plans list")
        return set()

    # Remove all white spaces from the end 
    invalid_plans = [line.rstrip() for line in invalid_plans]

    # Remove comments and empty strings 
    invalid_plans = [plan for plan in invalid_plans if plan != "" and not plan.startswith("#")]

    return set(invalid_plans)


def random_enum_vector(enum_class, size):
    return [random.choice(list(enum_class)) for _ in range(size)]


def data_augmentation(plans, config):
    new_plans = []
    
    for technique in config:
        for k, v in technique.items():
            cnt = int(v["percentage"] * len(plans))
            samples = random.sample(plans, cnt)

            if k == "permutation":
                new_plans += [permutate(plan) for plan in samples]

            elif k == "symmetry":
                sym_types = random_enum_vector(SymmetryType, len(samples))
                new_plans += [symmetry(plan, type) for plan, type in zip(samples, sym_types)]

            elif k == "rotation":
                rot_angles = random_enum_vector(RotationAngle, len(samples))
                new_plans += [rotation(plan, angle) for plan, angle in zip(samples, rot_angles)]

    plans = plans + new_plans
    random.shuffle(plans)

    return plans



def preprocess(file: str, plans):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, mode='w+') as file:
        for i, floor_plan in enumerate(plans):

            seq = to_sequence(floor_plan)

            for item in seq:
                file.write(item)

            file.write("\n")

            if (i+1) % 100 == 0 or i+1 == len(plans):
                print(f"{i+1}/{len(plans)}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Convert floor plans into sequences"
    )

    parser.add_argument(
        "path_to_config",
        type=str,
        help="Path to the configuration file"
    )

    parser.add_argument(
        "path_to_dataset",
        type=str,
        help="Paths to pre-processed RPLAN dataset file"
    )

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    with open(args.path_to_config, "r") as f:
        config = yaml.load(f, Loader=Loader)

    prep_config = config["preprocessing"]
    paths_config = config["paths"]

    data = sio.loadmat(args.path_to_dataset, squeeze_me=True, struct_as_record=False)['data']
    print("Dataset loaded")

    np.random.shuffle(data)

    floor_plans_cnt = len(data)
    if "max_floor_plans" in prep_config:
        floor_plans_cnt = min(floor_plans_cnt, prep_config["max_floor_plans"])

    data = data[0:floor_plans_cnt]

    invalid_plans = load_invalid()
    plans = [from_mat_file(plan) for plan in data if not plan.name in invalid_plans]

    if "data_augmentation" in prep_config:
        plans = data_augmentation(plans, prep_config["data_augmentation"])

    floor_plans_cnt = len(plans)

    validate_cnt = int(floor_plans_cnt * prep_config["validate_size"])
    test_cnt = int(floor_plans_cnt * prep_config["test_size"])
    train_cnt = floor_plans_cnt - validate_cnt - test_cnt

    
    print(f"Loaded {len(invalid_plans)} invalid plans")

    print("Preprocessing train dataset")
    preprocess(paths_config["input_data"] + "/train.txt", plans[0:train_cnt])

    print("Preprocessing test dataset")
    test_end_idx = test_cnt + train_cnt
    preprocess(paths_config["input_data"] + "/test.txt", plans[train_cnt: test_end_idx])

    print("Preprocessing validation dataset")
    preprocess(paths_config["input_data"] + "/validation.txt", plans[test_end_idx:floor_plans_cnt])


if __name__ == "__main__":
    main(sys.argv[1:])

