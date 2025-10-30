from transformers import AutoModelForCausalLM

import torch.nn as nn

from .gemma3 import get_gemma3
from .gpt2 import get_gpt2, get_gpt2_config
from .gpt2_with_coord_indices import GPT2ModelWithCoordIndices


def get_model(config, tokens_cnt) -> nn.Module:
    if config["type"] == "gemma3":
        return get_gemma3(config, tokens_cnt)
    
    elif config["type"] == "gpt2":
        if config["with_coord_indices"]:
            gpt2_config = get_gpt2_config(config, tokens_cnt)
            return GPT2ModelWithCoordIndices(gpt2_config)
        return get_gpt2(config, tokens_cnt)
    
    elif config["type"] == "existing":
        if config["with_coord_indices"]:
            return GPT2ModelWithCoordIndices.from_pretrained(config["input_model_path"])
        
        return AutoModelForCausalLM.from_pretrained(config["input_model_path"])
    
    else:
        raise ValueError("Invalid model type")
    

def print_model(model: nn.Module):
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
