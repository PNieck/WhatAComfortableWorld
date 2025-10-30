from __future__ import annotations
import argparse
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

    print("Starting training…")
    train(model, tokenizer, tokenized_dataset, train_config)

    print("Saving model")
    model.save_pretrained(paths_config["trained_model"])

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
