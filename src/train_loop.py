from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os

from tqdm.auto import tqdm

from transformers import DataCollatorForLanguageModeling

from src.log_writer import LogWriter
from src.losses import get_loss
from src.losses.neighborhood_loss import NeighborhoodLoss
from src.training_config import TrainingConfig
from src.lr_schedulers import get_lr_scheduler
from src.checkpoints import create_checkpoint, save_training_status, CheckpointReader


def calc_correct_preds(preds: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    preds_made = preds[:, :-1]
    labels_to_guess = labels[:, 1:]

    mask = labels_to_guess != -100
    
    correct = preds_made == labels_to_guess

    correct_guesses_cnt = correct[mask].sum()
    all_guesses_cnt = mask.sum()

    return (correct_guesses_cnt.item(), all_guesses_cnt.item())



def evaluate(model: nn.Module, test_loader: DataLoader, loss_fun, log_writer: LogWriter, step) -> tuple[float, float]:
    total_eval_loss = 0
    correct_preds = 0
    total_preds = 0

    total_ergo_loss = 0
    total_std_loss = 0
    total_output_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if model.device.type != "cpu":
                batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            labels = batch["labels"]

            total_eval_loss += loss_fun(outputs, labels)
            if isinstance(loss_fun, NeighborhoodLoss):
                total_ergo_loss += loss_fun.ergonomic_loss_output(outputs, labels)
                total_std_loss += loss_fun.std_loss(outputs, labels)
                total_output_loss += outputs.loss.item()

            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)

            (correct, preds) = calc_correct_preds(preds, labels)

            correct_preds += correct
            total_preds += preds
            
    eval_avg_loss = total_eval_loss / len(test_loader)
    accuracy = correct_preds / total_preds

    if isinstance(loss_fun, NeighborhoodLoss):
        avg_ergo_loss = total_ergo_loss / len(test_loader)
        avg_std_loss = total_std_loss / len(test_loader)
        avg_output_loss = total_output_loss / len(test_loader)

        log_writer.add_scalar("Eval avg ergo loss", avg_ergo_loss, step)
        log_writer.add_scalar("Eval avg std loss", avg_std_loss, step)
        log_writer.add_scalar("Eval avg output loss", avg_output_loss, step)

    return (eval_avg_loss, accuracy)


def checkpointing(model, optimizer, lr_scheduler, config: TrainingConfig, epoch, step, log_writer: LogWriter):
    if epoch % config.checkpointing_frequency == 0 and epoch != config.epochs_cnt:
        print("Creating a checkpoint")

        create_checkpoint(model, optimizer, lr_scheduler, config, epoch, step, log_writer)


def training_loop(model: nn.Module, tokenizer, dataset, config: TrainingConfig, log_writer: LogWriter):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(dataset["train"], batch_size=config.batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(dataset["test"], batch_size=config.batch_size, collate_fn=data_collator)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    loss_fun = get_loss(config, tokenizer, train_dataloader, device)
    num_epochs = config.epochs_cnt
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_lr_scheduler(config, optimizer, num_training_steps)
    eval_steps = config.eval_steps

    progress_bar = tqdm(range(num_training_steps))

    if config.use_checkpoint:
        ch = CheckpointReader(config.checkpoint_path, config.checkpoint_epoch)
        ch.load_optimizer(optimizer, device)
        ch.load_lr_scheduler(lr_scheduler, device)
        progress_bar.update(config.checkpoint_epoch * len(train_dataloader))
    

    train_loss = 0

    step = 0
    start_epoch = 1 if not config.use_checkpoint else config.checkpoint_epoch + 1
    model.train()
    for epoch in range(start_epoch, num_epochs+1):
        for batch in train_dataloader:
            if device.type != "cpu":
                batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch.pop('labels') # Remove labels since we want to compute the loss manually

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = loss_fun(outputs, labels)
            train_loss += loss.item()

            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            if step % eval_steps == 0:
                train_avg_loss = train_loss / eval_steps
                log_writer.add_scalar("Train avg loss", train_avg_loss, step)

                eval_avg_loss, accuracy = evaluate(model, test_dataloader, loss_fun, log_writer, step)
                model.train()

                log_writer.add_scalar("Eval avg loss", eval_avg_loss, step)
                log_writer.add_scalar("Eval accuracy", accuracy, step)
                log_writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], step)

                print(f"Epoch: {epoch}/{num_epochs}, step: {step + log_writer.start_step}/{num_training_steps}")
                print(f"\tAvg train loss: {train_avg_loss:.5f}, Avg eval loss: {eval_avg_loss:.5f}, eval_accuracy: {accuracy:.5f}")

            progress_bar.update(1)
            step += 1

            train_loss = 0

        checkpointing(model, optimizer, lr_scheduler, config, epoch, step, log_writer)

    eval_avg_loss, accuracy = evaluate(model, test_dataloader, loss_fun, log_writer, step)
    log_writer.add_scalar("Eval avg loss", eval_avg_loss, step)
    log_writer.add_scalar("Eval accuracy", accuracy, step)
    save_training_status(config.log_dir, log_writer, epoch, step)
