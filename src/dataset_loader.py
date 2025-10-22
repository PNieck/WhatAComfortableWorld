import scipy.io as sio

from datasets import load_dataset

from enum import IntFlag, auto


INVALID_FLOOR_PLANS_FILE = "data/invalid_floor_plans.txt"


class Split(IntFlag):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def load_floor_plans_dataset(path: str, splits: Split):
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


def _load_invalid() -> set[str]:
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



def load_dataset_from_mat_file(path: str, exclude_invalid=True):
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)['data']

    if exclude_invalid:
        invalid_plans = _load_invalid()
        data = [plan for plan in data if not plan.name in invalid_plans]

    return data
