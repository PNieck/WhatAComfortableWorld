import scipy.io as sio
import numpy as np

import sys
import argparse
import os
import json

import tokens


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
            x = floor_plan.rBoundary[idx][corner_idx, 0]
            y = floor_plan.rBoundary[idx][corner_idx, 1]

            if (not np.issubdtype(x, np.integer)) or (not np.issubdtype(y, np.integer)):
                raise Exception()
            # x = int(x)
            # y = int(y)

            file.write(tokens.coord_token(x))
            file.write(tokens.coord_token(y))

            tokens_added += 2

        file.write(tokens.room_token(room_label))
        tokens_added += 1

    return tokens_added


def main(argv):
    parser = argparse.ArgumentParser(
        description="Convert floor plans into sequences"
    )

    parser.add_argument(
        "path_to_dataset",
        type=str,
        help="Paths to pre-processed RPLAN dataset file"
    )

    parser.add_argument(
        "--result_file",
        default="data/sequences.txt",
        type=str,
        help="Name of the result file"
    )

    args = parser.parse_args(argv)

    data = sio.loadmat(args.path_to_dataset, squeeze_me=True, struct_as_record=False)['data']
    print(f"{len(data)} floor plans loaded")

    max_tokens = 0

    with open(args.result_file, mode='w') as file:
        for i, floor_plan in enumerate(data):
            print(f"{i}/{len(data)}")
            tokens_added = 0

            tokens_added += add_boundary_to_sequence(file, floor_plan)
            tokens_added += add_door_to_sequence(file, floor_plan)
            tokens_added += add_rooms_to_sequence(file, floor_plan)

            file.write("\n")

            if max_tokens < tokens_added:
                max_tokens = tokens_added

    metadata = {"max_seq_len": max_tokens}

    filename, _ = os.path.splitext(args.result_file)
    with open(filename + ".meta.json", mode="w") as file:
        json.dump(metadata, file)


if __name__ == "__main__":
    main(sys.argv[1:])

