import json
import os
from functools import singledispatch

from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel
)

import torch.nn as nn

from .custom_gpt2 import get_gpt2_config, CustomGPT2
from .gpt2_with_xy_indices import GPT2ModelWithXYIndices
from .gpt2_with_corners_indices import GPT2ModelWithCornerIndices

from src.training_config import TrainingConfig


def get_model(config: TrainingConfig) -> PreTrainedModel:
    if config.start_with_pretrained_model:
        return get_pretrained_model(config)
    
    model_config = get_gpt2_config(config)

    if config.model_has_corner_indices:
        return GPT2ModelWithCornerIndices(model_config)
    
    elif config.model_has_xy_indices:
        return GPT2ModelWithXYIndices(model_config)
    
    return CustomGPT2(model_config)


@singledispatch
def get_pretrained_model(c) -> PreTrainedModel:
    raise Exception(f"Unexpected argument type {type(c)}")


@get_pretrained_model.register(TrainingConfig)
def _1(config: TrainingConfig) -> PreTrainedModel:
    path = config.start_pretrained_model_path
    path = os.path.join(path, "model")

    return get_pretrained_model(path)
        

@get_pretrained_model.register(str)
def _2(path: str) -> PreTrainedModel:
    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as file:
        data = json.load(file)
        
    architecture = data["architectures"][0]

    match architecture:
        case "GPT2ModelWithXYIndices":
            return GPT2ModelWithXYIndices.from_pretrained(path)
        
        case "GPT2ModelWithCornerIndices":
            return GPT2ModelWithCornerIndices.from_pretrained(path)
        
        case "CustomGPT2":
            return CustomGPT2.from_pretrained(path)
        
        case _:
            return AutoModelForCausalLM.from_pretrained(path)


def print_model_size(model: nn.Module):
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
