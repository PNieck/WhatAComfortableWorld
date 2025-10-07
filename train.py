from __future__ import annotations
import argparse
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from datasets import load_dataset
import torch
from transformers import (
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

import floor_plan_tokenizer
from src.model import FloorPlanGenModel


def tokenize_function(examples, tokenizer: PreTrainedTokenizer, seq_len: int):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=seq_len,
        return_tensors="pt"
    )


def group_texts(examples, block_size: int):
    # Concatenate then split into fixed-size blocks
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


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

    tokenizer = floor_plan_tokenizer.FloorPlanTokenizer()

    print("Initializing model from scratch…")
    model = FloorPlanGenModel(model_config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    # Load dataset
    print("Loading datasets")
    train_dataset = load_dataset(
        "text",
        data_files=[paths_config["input_data"] + "/train.txt"]
    )

    test_dataset = load_dataset(
        "text",
        data_files=[paths_config["input_data"] + "/test.txt"]
    )


    # Tokenize
    train_tokenized = train_dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, model_config["max_seq_len"]),
        batched=True,
        remove_columns=["text"],
    )

    test_tokenized = test_dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, model_config["max_seq_len"]),
        batched=True,
        remove_columns=["text"],
    )

    # Group into contiguous blocks for causal LM
    grouped_train = train_tokenized.map(
        lambda ex: group_texts(ex, model_config["max_seq_len"]),
        batched=True,
    )

    grouped_test = test_tokenized.map(
        lambda ex: group_texts(ex, model_config["max_seq_len"]),
        batched=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=paths_config["output_data"],
        overwrite_output_dir=True,
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["batch_size"],
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        learning_rate=float(train_config["lr"]),
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        report_to="none",
        save_safetensors=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=grouped_train["train"],
        eval_dataset=grouped_test["train"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training…")
    trainer.train()

    print("Saving model")
    model.save(paths_config["trained_model"])

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
