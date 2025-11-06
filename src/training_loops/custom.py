from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from transformers import (
    DataCollatorForLanguageModeling,
    get_scheduler
)

from torch.utils.tensorboard import SummaryWriter


def calc_correct_preds(preds: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    preds_made = preds[:, :-1]
    labels_to_guess = labels[:, 1:]

    mask = labels_to_guess != -100
    
    correct = preds_made == labels_to_guess

    correct_guesses_cnt = correct[mask].sum()
    all_guesses_cnt = mask.sum()

    return (correct_guesses_cnt.item(), all_guesses_cnt.item())



def evaluate(model: nn.Module, test_loader: DataLoader) -> tuple[float, float]:
    total_eval_loss = 0
    correct_preds = 0
    total_preds = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if model.device.type != "cpu":
                batch = {k: v.to(model.device) for k, v in batch.items()}
            
            outputs = model(**batch)

            total_eval_loss += outputs.loss.item()

            logits = outputs.logits
            labels = batch["labels"]

            preds = torch.argmax(logits, dim=-1)

            (correct, preds) = calc_correct_preds(preds, labels)

            correct_preds += correct
            total_preds += preds
            
    eval_avg_loss = total_eval_loss / len(test_loader)
    accuracy = correct_preds / total_preds

    return (eval_avg_loss, accuracy)



def checkpointing(model, config, epoch):
    if "checkpointing_frequency" not in config:
        return
    
    if epoch % config["checkpointing_frequency"] == 0 and epoch != config["epochs"] and epoch != 0:
        print("Creating a checkpoint")
        
        dir = config["log_dir"]
        dir += f"checkpoints/epoch_{epoch}/"

        model.save_pretrained(dir)



def custom_training_loop(model: nn.Module, tokenizer, dataset, config, tb: SummaryWriter):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        dataset["train"], batch_size=config["batch_size"], collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        dataset["test"], batch_size=config["batch_size"], collate_fn=data_collator
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    num_epochs = config["epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    eval_steps = config["eval_steps"]

    train_loss = 0

    step = 0
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            if device.type != "cpu":
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

                eval_avg_loss, accuracy = evaluate(model, test_dataloader)
                model.train()

                tb.add_scalar("Eval avg loss", eval_avg_loss, step)
                tb.add_scalar("Eval accuracy", accuracy, step)
                tb.add_scalar("lr", lr_scheduler.get_last_lr()[0], step)

                print(f"Epoch: {epoch}/{num_epochs}, step: {step}/{num_training_steps}")
                print(f"\tAvg train loss: {train_avg_loss:.5f}, Avg eval loss: {eval_avg_loss:.5f}, eval_accuracy: {accuracy:.5f}")

            progress_bar.update(1)
            step += 1

            train_loss = 0

        checkpointing(model, config, epoch)


    eval_avg_loss, accuracy = evaluate(model, test_dataloader)
    tb.add_scalar("Eval avg loss", eval_avg_loss, step)
    tb.add_scalar("Eval accuracy", accuracy, step)
