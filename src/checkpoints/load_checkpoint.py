import os
import json
from pathlib import Path
import re

import torch

from src.training_config import TrainingConfig
from src.log_writer import LogWriter
from src.models import get_pretrained_model


class CheckpointReader:
    def __init__(self, log_dir: str, epoch: int|None = None) -> None:
        dir = os.path.join(log_dir, "checkpoints")

        if not os.path.exists(dir):
            raise ValueError(f"No checkpoints dir in {log_dir}")

        if epoch is None:
            epoch = CheckpointReader._get_biggest_checkpoint_epoch(dir)

        self.dir: str = os.path.join(dir, f"epoch_{epoch}")
        if not os.path.exists(self.dir):
            raise ValueError(f"No directory {self.dir}")


    def load_model(self):
        return get_pretrained_model(self.dir)


    def load_optimizer(self, optimizer, device):
        optimizer_file = os.path.join(self.dir, "optimizer_state.pt")
        state = torch.load(optimizer_file, map_location=device)
        optimizer.load_state_dict(state)


    def load_lr_scheduler(self, scheduler, device):
        scheduler_file = os.path.join(self.dir, "scheduler.pt")
        state = torch.load(scheduler_file, map_location=device)
        scheduler.load_state_dict(state)


    def load_log_writer(self, log_writer: LogWriter):
        log_writer_file = os.path.join(self.dir, "log_writer.json")
        with open(log_writer_file, "r") as file:
            state = json.load(file)

        log_writer.load_state_dict(state)


    def load_training_config(self, config: TrainingConfig):
        config_file = os.path.join(self.dir, "training_config.json")
        with open(config_file, "r") as file:
            state = json.load(file)

        config.load_state_dict(state)


    @staticmethod
    def _get_biggest_checkpoint_epoch(dir: str) -> int:
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
