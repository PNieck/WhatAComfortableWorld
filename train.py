from __future__ import annotations
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

import floor_plan_tokenizer


def build_model(vocab_size: int, n_layer: int, n_head: int, n_embd: int, max_position: int) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_position,
        n_ctx=max_position,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        bos_token_id=0,  # will be set by tokenizer when resized
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    return model


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
    p = argparse.ArgumentParser(description="Train a tiny LLM from scratch")

    p.add_argument("--data_file", type=str, help="File with dataset", default="./data/sequences.txt")
    p.add_argument("--out_dir", type=str, default="./models")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)

    # Model size (start tiny, then scale up)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--checkpointing", action="store_true", help="Enable gradient checkpointing")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
    p.add_argument("--fp16", action="store_true", help="Use float16 mixed precision")
    p.add_argument("--push_to_hub", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)

    data_file = args.data_file
    out_dir = Path(args.out_dir)

    tokenizer = floor_plan_tokenizer.FloorPlanTokenizer()

    print("Initializing model from scratch…")
    model = build_model(
        vocab_size=len(tokenizer),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_position=args.seq_len,
    )

    # Tie tokenizer + model vocab sizes
    model.resize_token_embeddings(len(tokenizer))

    if args.checkpointing:
        model.gradient_checkpointing_enable()

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    # Load dataset
    print("Loading dataset…")
    dataset = load_dataset("text", data_files=[data_file])
    # Split train/validation (90/10)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)

    # Tokenize
    tokenized = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, args.seq_len),
        batched=True,
        remove_columns=["text"],
    )

    # Group into contiguous blocks for causal LM
    lm_datasets = tokenized.map(
        lambda ex: group_texts(ex, args.seq_len),
        batched=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = args.fp16 and torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_accumulation_steps=1,
        report_to="none",
        push_to_hub=args.push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training…")
    trainer.train()

    print("Saving model")
    trainer.save_model(out_dir)

    # Quick test generation
    prompt = "<Bound><Coord 193><Coord 103><Coord 197><Coord 103><Coord 197><Coord 208><Coord 153><Coord 208><Coord 153><Coord 180><Coord 60><Coord 180><Coord 60><Coord 75><Coord 68><Coord 75><Coord 68><Coord 78><Coord 134><Coord 78><Coord 134><Coord 51><Coord 197><Coord 51><Coord 197><Coord 73><Coord 193><Coord 73><Bound>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
        )
    print("\n=== Sample generation ===\n")
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    print(f"\nAll done. Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
