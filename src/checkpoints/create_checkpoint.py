"""Checkpoint creation utilities for saving training state.

Provides functions for serializing model, optimizer, and scheduler state to disk
for training resumption.
"""

import os
import json

import torch

from src.training_config import TrainingConfig
from src.log_writer import LogWriter


def create_checkpoint(model, optimizer, lr_scheduler, config: TrainingConfig, epoch, step, log_writer: LogWriter):
    """Save current training state to checkpoint directory.

    Serializes model weights, optimizer state, scheduler state, and training metadata
    to disk, allowing training to be resumed from this point.
    
    :param model: Neural network model to save
    :param optimizer: Optimizer with current parameter states
    :param lr_scheduler: Learning rate scheduler with current state
    :param config: Training configuration object
    :param epoch: Current training epoch
    :param step: Current training step
    :param log_writer: Logger object for training status
    """

    dir = os.path.join(config.log_dir, "checkpoints", f"epoch_{epoch}")
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
    """
    Saves information for how many epochs and steps was done during training
    
    :param dir: Directory where to save a `training_status.json` file
    :type dir: str
    :param log_writer: A logging object
    :type log_writer: LogWriter
    :param epoch: The current training epoch
    :param step: The current training step
    """

    state = log_writer.state_dict(epoch, step)
    file = os.path.join(dir, "training_status.json")
    _save_to_json(state, file)


def _save_to_json(dict, path):
    """
    Saves python dictionary as a json file
    
    :param dict: python dictionary
    :param path: path containing json file to which dictionary is going to be saved
    """

    with open(path, "w") as file:
        file.write(json.dumps(dict, indent=4))
