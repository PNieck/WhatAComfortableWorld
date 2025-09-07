import scipy.io as sio
import numpy as np

import sys
import argparse
import os
import json
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import tokens


INVALID_FLOOR_PLANS_FILE = "data/invalid_floor_plans.txt"


max_tokens = 0


def load_invalid() -> set:
    try:
        with open(INVALID_FLOOR_PLANS_FILE, mode='r') as file:
            invalid_plans = file.readlines()
    except FileNotFoundError:
        print("WARNING: cannot load invalid floor plans list")
        return set()

    # Remove all white spaces from the end 
    invalid_plans = [line.rstrip() for line in invalid_plans]

    return set(invalid_plans)


def add_boundary_coord(index, floor_plan, file):
    x = floor_plan.boundary[index, 0]
    y = floor_plan.boundary[index, 1]

    file.write(tokens.coord_token(x))
    file.write(tokens.coord_token(y))


def add_door_to_sequence(file, floor_plan) -> int:
    file.write(tokens.DOOR_TOKEN)

    add_boundary_coord(0, floor_plan, file)
    add_boundary_coord(1, floor_plan, file)

    file.write(tokens.DOOR_TOKEN)

    return 6


def add_boundary_to_sequence(file, floor_plan) -> int:
    file.write(tokens.BOUNDARY_TOKEN)
    tokens_added = 1

    if floor_plan.boundary[0, 3] == 0:
        add_boundary_coord(0, floor_plan, file)
        tokens_added += 2

    if floor_plan.boundary[1, 3] == 0:
        add_boundary_coord(1, floor_plan, file)
        tokens_added += 2

    for i in range(2, floor_plan.boundary.shape[0]):
        add_boundary_coord(i, floor_plan, file)
        tokens_added += 2

    file.write(tokens.BOUNDARY_TOKEN)

    return tokens_added + 1


def add_rooms_to_sequence(file, floor_plan):
    tokens_added = 0

    for idx, room_label in enumerate(floor_plan.rType):
        file.write(tokens.room_token(room_label))
        tokens_added += 1

        for corner_idx in range(floor_plan.rBoundary[idx].shape[0]):
            x = int(floor_plan.rBoundary[idx][corner_idx, 0])
            y = int(floor_plan.rBoundary[idx][corner_idx, 1])

            file.write(tokens.coord_token(x))
            file.write(tokens.coord_token(y))

            tokens_added += 2

        file.write(tokens.room_token(room_label))
        tokens_added += 1

    return tokens_added


def preprocess(file: str, start_idx: int, end_idx: int, data: np.ndarray, invalid_plans: set):
    global max_tokens

    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, mode='w+') as file:
        for i, floor_plan in enumerate(data[start_idx:end_idx]):
            tokens_added = 0

            if floor_plan.name in invalid_plans:
                print(f"Skipping invalid file {floor_plan.name}")
                continue

            tokens_added += add_boundary_to_sequence(file, floor_plan)
            tokens_added += add_door_to_sequence(file, floor_plan)
            tokens_added += add_rooms_to_sequence(file, floor_plan)

            file.write("\n")

            if (i+1) % 100 == 0 or i+1 == end_idx-start_idx:
                print(f"{i+1}/{end_idx-start_idx}")

            if max_tokens < tokens_added:
                max_tokens = tokens_added


def main(argv):
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

    args = parser.parse_args(argv)

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

    validate_cnt = int(floor_plans_cnt * prep_config["validate_size"])
    test_cnt = int(floor_plans_cnt * prep_config["test_size"])
    train_cnt = floor_plans_cnt - validate_cnt - test_cnt

    invalid_plans = load_invalid()

    print("Preprocessing train dataset")
    preprocess(paths_config["input_data"] + "/train.txt", 0, train_cnt, data, invalid_plans)

    print("Preprocessing test dataset")
    test_end_idx = test_cnt + train_cnt
    preprocess(paths_config["input_data"] + "/test.txt", train_cnt, test_end_idx, data, invalid_plans)

    print("Preprocessing validation dataset")
    preprocess(paths_config["input_data"] + "/validation.txt", test_end_idx, floor_plans_cnt, data, invalid_plans)

    metadata = {"max_seq_len": max_tokens}
    filename, _ = os.path.splitext(paths_config["input_data"] + "/metadata.json")
    with open(filename + ".meta.json", mode="w+") as file:
        json.dump(metadata, file)


if __name__ == "__main__":
    main(sys.argv[1:])

