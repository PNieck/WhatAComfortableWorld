from __future__ import annotations
import argparse
import os

from transformers import (
    PreTrainedTokenizer,
    set_seed,
)

from src.models import (
    print_model_size,
    get_model
)

from src.train_loop import training_loop
from src.dataset_loader import load_floor_plans_dataset, Split
from src.floor_plan_tokenizer import FloorPlanTokenizer
from src.validation import validate
from src.log_writer import LogWriter
from src.training_config import TrainingConfig


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

    config = TrainingConfig(args.path_to_config)

    if config.seed is not None:
        set_seed(config.seed)

    tokenizer = FloorPlanTokenizer()
    config.update_with_tokenizer(tokenizer)

    model = get_model(config)
    
    print_model_size(model)
    print(model)

    # Load dataset
    print("Loading datasets")
    dataset = load_floor_plans_dataset(config.input_data_path, Split.TEST | Split.TRAIN)

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, config.max_sequence_len),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_dataset.set_format("torch")

    log_writer = LogWriter(config.log_dir)

    print("Starting training…")
    training_loop(model, tokenizer, tokenized_dataset, config, log_writer)

    print("Saving model")
    model_dir = os.path.join(config.log_dir, "model")
    model.save_pretrained(model_dir)

    print("Validating model…")
    dataset = load_floor_plans_dataset(config.input_data_path, Split.VALID)
    validate(model, tokenizer, dataset, config, log_writer)

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
