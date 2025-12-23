from __future__ import annotations
import argparse
import datetime
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from transformers import (
    PreTrainedTokenizer,
    set_seed,
)

from src.models import (
    print_model_size,
    get_model,
    preprocess_model_config
)

from src.train_loop import train
from src.dataset_loader import load_floor_plans_dataset, Split
from src.floor_plan_tokenizer import FloorPlanTokenizer
from src.validation import validate
from src.log_writer import LogWriter


def log_dir_name(training_config: dict, model_config: dict) -> str:
    dir_prefix = "runs/"
    
    if "input_model_path" in model_config:
        result = dir_prefix + model_config["input_model_path"]
        if not result.endswith("/"):
            result += "/"
        
        return result

    result = dir_prefix + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    if "log_comment" in training_config:
        result += training_config["log_comment"] + "/"

    return result


def tokenize_function(examples, tokenizer: PreTrainedTokenizer, seq_len: int):
    return tokenizer(
        examples["text"],
        padding=False,      
        truncation=True,
        max_length=seq_len,
    )
    

def main():
    p = argparse.ArgumentParser(description="Train model from scratch")

    p.add_argument(
        "path_to_config",
        type=str,
        help="Path to the configuration file"
    )
    args = p.parse_args()

    with open(args.path_to_config, "r") as f:
        config = yaml.load(f, Loader=Loader)

    model_config = config["model"]
    paths_config = config["paths"]
    train_config = config["training"]

    if "seed" in config["general"]:
        set_seed(config["general"]["seed"])

    tokenizer = FloorPlanTokenizer()

    model_config = preprocess_model_config(model_config, tokenizer)
    model = get_model(model_config)
    
    print_model_size(model)
    print(model)

    # Load dataset
    print("Loading datasets")
    dataset = load_floor_plans_dataset(paths_config["input_data"], Split.TEST | Split.TRAIN)

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, model_config["max_seq_len"]),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_dataset.set_format("torch")

    log_dir = log_dir_name(train_config, model_config)
    train_config["log_dir"] = log_dir
    log_writer = LogWriter(log_dir)

    print("Starting training…")
    train(model, tokenizer, tokenized_dataset, train_config, log_writer)

    print("Saving model")
    model.save_pretrained(log_dir + "model/")

    print("Validating model…")
    dataset = load_floor_plans_dataset(paths_config["input_data"], Split.VALID)
    validate(model, tokenizer, dataset, train_config, model_config, log_writer)

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
