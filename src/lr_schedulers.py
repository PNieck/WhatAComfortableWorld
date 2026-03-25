"""Learning rate scheduler factory utilities.

Provides factory function for creating learning rate schedulers based on
training configuration (linear or cyclic).
"""

import torch

from transformers import get_scheduler

from src.training_config import TrainingConfig


def get_lr_scheduler(config: TrainingConfig, optimizer, training_steps: int):
    """Create learning rate scheduler based on training configuration.
    
    Supports linear and cyclic learning rate schedules configured through
    the training configuration object.
    
    :param config: Training configuration specifying scheduler type and parameters
    :param optimizer: Optimizer to apply scheduler to
    :param training_steps: Total number of training steps
    :return: Learning rate scheduler object
    :raises ValueError: If scheduler type is not recognized
    """

    match config.lr_scheduler_type:
        case "linear":
            return get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=training_steps,
            )
        
        case "cyclic":
            scheduler_config = config.lr_scheduler_config
            assert scheduler_config is not None

            return torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=float(scheduler_config.base_lr),
                max_lr=float(scheduler_config.max_lr),
                step_size_up=training_steps // (2*int(scheduler_config.cycles))
            )
        
        case _:
            raise ValueError(f"Unexpected lr scheduler type {config.lr_scheduler_type}")

        
