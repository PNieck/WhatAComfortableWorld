"""Checkpoint loading utilities for resuming training from checkpoints.

Provides CheckpointReader class for loading saved model, optimizer, scheduler,
and training state from checkpoint files.
"""

import os
import json
from pathlib import Path
import re

import torch

from src.training_config import TrainingConfig
from src.log_writer import LogWriter
from src.models import get_pretrained_model


class CheckpointReader:
    """Reader for loading training checkpoints and their associated state.
    
    Loads model, optimizer, scheduler, and training status from checkpoint files
    created by create_checkpoint function.
    """

    def __init__(self, log_dir: str, epoch: int|None = None) -> None:
        """Initialize CheckpointReader for accessing saved training state.
        
        :param log_dir: Directory containing checkpoint subdirectory
        :param epoch: Specific checkpoint epoch to load. If None, loads latest (default: None)
        :raises ValueError: If checkpoints directory or specified epoch not found
        """

        dir = os.path.join(log_dir, "checkpoints")

        if not os.path.exists(dir):
            raise ValueError(f"No checkpoints dir in {log_dir}")

        self.epoch = epoch

        if self.epoch is None:
            self.epoch = CheckpointReader._get_biggest_checkpoint_epoch(dir)

        self.dir: str = os.path.join(dir, f"epoch_{self.epoch}")
        if not os.path.exists(self.dir):
            raise ValueError(f"No directory {self.dir}")


    def load_model(self):
        """
        Loads the pre-trained model from the checkpoint directory.
        
        :return: The loaded pre-trained model
        """

        return get_pretrained_model(self.dir)


    def load_optimizer(self, optimizer, device):
        """
        Loads the optimizer state from the checkpoint into the provided optimizer.
        
        :param optimizer: The optimizer to load state into
        :param device: The device to load the optimizer state onto
        """

        optimizer_file = os.path.join(self.dir, "optimizer_state.pt")
        state = torch.load(optimizer_file, map_location=device)
        optimizer.load_state_dict(state)


    def load_lr_scheduler(self, scheduler, device):
        """
        Loads the learning rate scheduler state from the checkpoint into the provided scheduler.
        
        :param scheduler: The scheduler to load state into
        :param device: The device to load the scheduler state onto
        """

        scheduler_file = os.path.join(self.dir, "scheduler.pt")
        state = torch.load(scheduler_file, map_location=device)
        scheduler.load_state_dict(state)


    def load_log_writer(self, log_writer: LogWriter):
        """
        Loads the log writer state from the checkpoint into the provided log writer.
        
        :param log_writer: The log writer to load state into
        :type log_writer: LogWriter
        """

        log_writer_file = os.path.join(self.dir, "log_writer.json")
        with open(log_writer_file, "r") as file:
            state = json.load(file)

        log_writer.load_state_dict(state)


    def load_training_config(self, config: TrainingConfig):
        """
        Loads the training configuration state from the checkpoint into the provided config.
        
        :param config: The training config to load state into
        :type config: TrainingConfig
        """

        config_file = os.path.join(self.dir, "training_config.json")
        with open(config_file, "r") as file:
            state = json.load(file)

        config.load_state_dict(state)


    @staticmethod
    def _get_biggest_checkpoint_epoch(dir: str) -> int:
        """
        Finds and returns the largest checkpoint epoch number in the given directory.
        
        Scans the directory for subdirectories matching the pattern 'epoch_*' and returns
        the highest epoch number found.
        
        :param dir: The directory to scan for checkpoints
        :type dir: str
        :return: The largest epoch number found
        :rtype: int
        :raises ValueError: If no valid checkpoint directories are found
        """
        
        tmp_dir = Path(dir)

        max_epoch = -1
        for entry in tmp_dir.iterdir():
            if not entry.is_dir():
                continue

            match = re.match(r"^epoch_(\d+)", entry.name)
            if not match:
                continue

            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch

        if max_epoch < 0:
            raise ValueError(f"Invalid checkpoint dir {dir}")

        return max_epoch
