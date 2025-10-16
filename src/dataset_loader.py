from datasets import load_dataset


def load_floor_plans_dataset(path: str):
    dataset = load_dataset(
        "text",
        data_files=[path + "/train.txt"]
    )

    dataset["test"] = load_dataset(
        "text",
        data_files=[path + "/test.txt"]
    )["train"]

    return dataset