from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel
)

import torch.nn as nn

from .gemma3 import get_gemma3
from .gpt2 import get_gpt2, get_gpt2_config
from .gpt2_with_coord_indices import GPT2ModelWithXYIndices
from .gpt2_with_corners_indices import GPT2ModelWithCornerIndices


def get_model(config) -> nn.Module:
    if config["type"] == "gemma3":
        return get_gemma3(config)
    
    elif config["type"] == "gpt2":
        if config["with_corner_indices"]:
            gpt2_config = get_gpt2_config(config)
            return GPT2ModelWithCornerIndices(gpt2_config)

        if config["with_coord_indices"]:
            gpt2_config = get_gpt2_config(config)
            return GPT2ModelWithXYIndices(gpt2_config)
        
        return get_gpt2(config)
    
    elif config["type"] == "existing":
        if config["with_coord_indices"]:
            return GPT2ModelWithXYIndices.from_pretrained(config["input_model_path"])
        
        return AutoModelForCausalLM.from_pretrained(config["input_model_path"])
    
    else:
        raise ValueError("Invalid model type")
    

def get_pretrained_model(path, config) -> PreTrainedModel:
    if config["with_corner_indices"]:
        return GPT2ModelWithCornerIndices.from_pretrained(path)

    if config["with_coord_indices"]:
        return GPT2ModelWithXYIndices.from_pretrained(path)
    else:
        return AutoModelForCausalLM.from_pretrained(path)
    

def print_model_size(model: nn.Module):
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
