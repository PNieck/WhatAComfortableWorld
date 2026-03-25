"""TensorBoard logging utilities for training and evaluation.

Provides LogWriter class for writing scalars and hyperparameters to TensorBoard,
with support for checkpoint-based resumption of training.
"""

from torch.utils.tensorboard import SummaryWriter

import os
import json


class LogWriter:
    """Logger for writing metrics to TensorBoard.
    
    Wraps TensorBoard SummaryWriter and provides checkpoint support for resuming
    training from a specific epoch and step.
    """

    def __init__(self, log_dir: str):
        """Initialize LogWriter with a directory for logs.
        
        :param log_dir: Directory to store TensorBoard logs
        """
        self.start_epoch = 0
        self.start_step = 0

        self.sw = SummaryWriter(log_dir)

    @classmethod
    def from_checkpoint(cls, log_dir, checkpoint_epoch):
        """Create LogWriter from a checkpoint for resuming training.
        
        :param log_dir: Log directory containing checkpoints
        :param checkpoint_epoch: Epoch number of checkpoint to load
        :return: LogWriter instance with state restored from checkpoint
        :raises Exception: If checkpoint directory or training status file not found
        """
        result = LogWriter(log_dir)

        checkpoint_dir = os.path.join(log_dir, "checkpoints", f"epoch_{checkpoint_epoch}")
        if not os.path.isdir(checkpoint_dir):
            raise Exception(f"No checkpoint dir {checkpoint_dir}")
        
        training_status_path =  os.path.join(checkpoint_dir, "training_status.json")
        if not os.path.exists(training_status_path):
            raise Exception(f"No training status file {training_status_path}")
        
        with open(training_status_path, "r") as file:
            training_status = json.load(file)
            result.start_epoch = training_status.get("epochs")
            result.start_step = training_status.get("steps")

        return result


    def add_scalar(self, tag: str, scalar_value: float, step: int):
        """Log a scalar value to TensorBoard.
        
        :param tag: Tag name for the scalar
        :param scalar_value: Value to log
        :param step: Step number (adjusted by start_step for resumed training)
        """
        self.sw.add_scalar(tag, scalar_value, step + self.start_step)


    def add_hparams(self, hparam_dict: dict, metric_dict: dict):
        """Log hyperparameters and final metrics to TensorBoard.
        
        :param hparam_dict: Dictionary of hyperparameters
        :param metric_dict: Dictionary of final metric values
        """
        self.sw.add_hparams(hparam_dict, metric_dict)


    def state_dict(self, epoch, step) -> dict:
        """Get current training state for checkpointing.
        
        :param epoch: Current epoch number
        :param step: Current step number
        :return: Dictionary with epoch and adjusted step
        """
        return {
            "epochs": epoch,
            "steps": step + self.start_step
        }
    
    def load_state_dict(self, dict: dict):
        """Load training state from dictionary.
        
        :param dict: State dictionary with "epochs" and "steps" keys
        """
        self.start_epoch = dict["epochs"]
        self.start_step = dict["steps"]

