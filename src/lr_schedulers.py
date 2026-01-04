import torch

from transformers import get_scheduler

from src.training_config import TrainingConfig


def get_lr_scheduler(config: TrainingConfig, optimizer, training_steps: int):
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

        
