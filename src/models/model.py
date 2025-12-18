import json
from pathlib import Path
import re

from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel
)

import torch.nn as nn

from .gemma3 import get_gemma3
from .custom_gpt2 import get_gpt2, get_gpt2_config, CustomGPT2
from .gpt2_with_xy_indices import GPT2ModelWithXYIndices
from .gpt2_with_corners_indices import GPT2ModelWithCornerIndices


def get_model(config) -> nn.Module:
    if "input_model_path" in config:
        return get_pretrained_model(config)

    if config["type"] == "gemma3":
        return get_gemma3(config)
    
    elif config["type"] == "gpt2":
        if config["with_corner_indices"]:
            gpt2_config = get_gpt2_config(config)
            return GPT2ModelWithCornerIndices(gpt2_config)

        if config["with_xy_indices"]:
            gpt2_config = get_gpt2_config(config)
            return GPT2ModelWithXYIndices(gpt2_config)
        
        return get_gpt2(config)
    
    else:
        raise ValueError("Invalid model type")
    

def get_pretrained_model(config) -> PreTrainedModel:
    path = "runs/" + config["input_model_path"]

    if config["from_checkpoint"]:
        path = _get_checkpoint_dir(config, path)
    else:
        path += "/model"

    with open(path + "/config.json", "r") as file:
        data = json.load(file)
        
    architecture = data["architectures"][0]

    if architecture == "GPT2ModelWithXYIndices":
        return GPT2ModelWithXYIndices.from_pretrained(path)
    elif architecture == "GPT2ModelWithCornerIndices":
        return GPT2ModelWithCornerIndices.from_pretrained(path)
    elif architecture == "CustomGPT2":
        return CustomGPT2.from_pretrained(path)
    else:
        return AutoModelForCausalLM.from_pretrained(path)
    

def _get_checkpoint_dir(config, path) -> str:
    if "checkpoint_epoch" in config:
        epoch_number = config["checkpoint_epoch"]
        
    else:
        tmp_path = Path(path + "/checkpoints")

        max_epoch = -1
        for entry in tmp_path.iterdir():
            if not entry.is_dir():
                continue

            match = re.match(r"^epoch_(\d+)", entry.name)
            if not match:
                continue

            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch

        epoch_number = max_epoch

    return path + f"/checkpoints/epoch_{epoch_number}"
        

def print_model_size(model: nn.Module):
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
