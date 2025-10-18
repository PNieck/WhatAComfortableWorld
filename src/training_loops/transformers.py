import torch.nn as nn
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def transformers_training_loop(model: nn.Module, tokenizer, dataset, config):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        logging_steps=2,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        learning_rate=float(config["lr"]),
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        report_to="tensorboard",
        logging_dir="runs/"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training…")
    trainer.train()
