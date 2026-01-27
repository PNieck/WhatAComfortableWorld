import scipy.io as sio

from datasets import load_dataset, DatasetDict

from enum import IntFlag, auto


INVALID_FLOOR_PLANS_FILE = "data/invalid_floor_plans.txt"
IMPERFECT_FLOOR_PLANS_FILE = "data/imperfect_floor_plans.txt"


class Split(IntFlag):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def load_floor_plans_dataset(path: str, splits: Split = Split.TRAIN | Split.VALID | Split.TEST) -> DatasetDict:
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


def _load_invalid() -> set[str]:
    return _load_floor_plan_names(INVALID_FLOOR_PLANS_FILE)


def load_imperfect_floor_plans_names() -> set[str]:
    return _load_floor_plan_names(IMPERFECT_FLOOR_PLANS_FILE)


def load_dataset_from_mat_file(path: str, exclude_invalid=True, exclude_imperfect=False):
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)['data']

    if exclude_invalid:
        invalid_plans = _load_invalid()
        data = [plan for plan in data if not plan.name in invalid_plans]
    
    if exclude_imperfect:
        imperfect_plans = load_imperfect_floor_plans_names()
        data = [plan for plan in data if not plan.name in imperfect_plans]

    return data
