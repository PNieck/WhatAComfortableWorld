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
from src.checkpoints import CheckpointReader


def tokenize_function(examples, tokenizer: PreTrainedTokenizer, seq_len: int):
    return tokenizer(
        examples["text"],
        padding=False,      
        truncation=True,
        max_length=seq_len,
    )
    

def main():
    p = argparse.ArgumentParser(description="Train model from scratch")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        dest="config",
        type=str,
        help="Path to the configuration file"
    )

    group.add_argument(
        "--from_checkpoint",
        dest="from_checkpoint",
        type=str,
        help="Path to a checkpoint directory (e.g. runs/.../checkpoints/epoch_3)"
    )

    p.add_argument(
        "--epoch",
        dest="epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to resume from (overrides epoch detected from directory). Only valid with --from_checkpoint."
    )

    args = p.parse_args()

    tokenizer = FloorPlanTokenizer()

    if args.config is not None:
        config = TrainingConfig(args.config)
        config.update_with_tokenizer(tokenizer)
        model = get_model(config)
        log_writer = LogWriter(config.log_dir)

    else:
        ch = CheckpointReader(args.from_checkpoint, args.epoch)
        
        config = TrainingConfig()
        config.update_with_tokenizer(tokenizer)
        ch.load_training_config(config)
        config.checkpoint_path = args.from_checkpoint
        config.checkpoint_epoch = ch.epoch

        log_writer = LogWriter(config.log_dir)
        ch.load_log_writer(log_writer)

        model = ch.load_model()

    if config.seed is not None:
        set_seed(config.seed)
    
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
