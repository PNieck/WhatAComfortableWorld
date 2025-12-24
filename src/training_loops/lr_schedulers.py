import torch

from transformers import get_scheduler


def get_lr_scheduler(config, optimizer, training_steps: int):
    if "lr_scheduler" not in config:
        print("Learning scheduler not specified - using linear one")
        config["lr_scheduler"]["type"] = "linear"

    scheduler_config = config["lr_scheduler"]

    if scheduler_config["type"] == "linear":
        return get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=training_steps,
        )
    
    if scheduler_config["type"] == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=float(scheduler_config["base_lr"]),
            max_lr=float(scheduler_config["max_lr"]),
            step_size_up=training_steps // (2*int(scheduler_config["cycles"]))
        )
