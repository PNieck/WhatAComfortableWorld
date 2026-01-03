import os
import json

import torch

from src.training_config import TrainingConfig
from src.log_writer import LogWriter


def create_checkpoint(model, optimizer, lr_scheduler, config: TrainingConfig, epoch, step, log_writer: LogWriter):
    dir = os.path.join(config.log_dir, "checkpoints", f"epoch_{epoch + log_writer.start_epoch}")
    os.makedirs(dir, exist_ok=True)

    model.save_pretrained(dir)

    state = optimizer.state_dict()
    optimizer_file = os.path.join(dir, "optimizer_state.pt")
    torch.save(state, optimizer_file)

    state = lr_scheduler.state_dict()
    scheduler_file = os.path.join(dir, "scheduler.pt")
    torch.save(state, scheduler_file)

    state = log_writer.state_dict(epoch, step)
    log_writer_file = os.path.join(dir, "log_writer.json")
    _save_to_json(state, log_writer_file)

    state = config.state_dict()
    config_file = os.path.join(dir, "training_config.json")
    _save_to_json(state, config_file)


def save_training_status(dir: str, log_writer: LogWriter, epoch, step):
    state = log_writer.state_dict(epoch, step)
    file = os.path.join(dir, "training_status.json")
    _save_to_json(state, file)


def _save_to_json(dict, path):
    with open(path, "w") as file:
        file.write(json.dumps(dict, indent=4))
