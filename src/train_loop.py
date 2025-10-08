from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from transformers import (
    DataCollatorForLanguageModeling,
    get_scheduler
)

from torch.utils.tensorboard import SummaryWriter

import tokens



def evaluate(model: nn.Module, test_loader: DataLoader, device, tb: SummaryWriter, step) -> tuple[float, float]:
    total_eval_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)

            total_eval_loss += outputs.loss.item()

            logits = outputs.logits
            labels = batch["labels"]

            preds = torch.argmax(logits, dim=-1)
            mask = labels != tokens.PAD_TOKEN_ID
            correct = (preds == labels) & mask

            correct_preds += correct.sum().item()
            total_preds += mask.sum().item()
            
    eval_avg_loss = total_eval_loss / len(test_loader)
    accuracy = correct_preds / total_preds

    tb.add_scalar("Eval avg loss", eval_avg_loss, step)
    tb.add_scalar("Eval precision", accuracy, step)

    return (eval_avg_loss, accuracy)



def train(model: nn.Module, tokenizer, dataset, train_config):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        dataset["train"], batch_size=train_config["batch_size"], collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        dataset["test"], batch_size=train_config["batch_size"], collate_fn=data_collator
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    num_epochs = train_config["epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    tb = SummaryWriter(comment=train_config["log_comment"])

    eval_steps = train_config["eval_steps"]

    train_loss = 0

    step = 0
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            if step % eval_steps == 0:
                train_avg_loss = train_loss / eval_steps
                tb.add_scalar("Train avg loss", train_avg_loss, step)

                eval_avg_loss, accuracy = evaluate(model, test_dataloader, device, tb, step)

                print(f"Epoch: {epoch}/{num_epochs}, step: {step}/{num_training_steps}")
                print(f"\tAvg train loss: {train_avg_loss:.3f}, Avg eval loss: {eval_avg_loss:.3f}, eval_accuracy: {accuracy:.3f}")

            progress_bar.update(1)
            step += 1

            train_loss = 0

    evaluate(model, test_dataloader, device, tb, step)

