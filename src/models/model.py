"""Model factory and loader utilities for floor plan generation.

Provides functions for creating and loading models with various configurations
including custom GPT-2 variants with spatial coordinate indices.
"""

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
from .gpt2_with_vertex_indices import GPT2ModelWithVertexIndices

from src.training_config import TrainingConfig


def get_model(config: TrainingConfig) -> PreTrainedModel:
    """Get or create a model based on training configuration.
    
    Loads pre-trained model if specified in config, otherwise creates
    a new model based on configuration parameters.
    
    :param config: Training configuration
    :return: Model instance
    :rtype: PreTrainedModel
    """

    if config.start_with_pretrained_model:
        return get_pretrained_model(config)
    
    model_config = get_gpt2_config(config)

    if config.model_has_vertex_indices:
        model = GPT2ModelWithVertexIndices(model_config)

        if not config.model_has_xy_indices:
            model.use_xy_indices = False

        return model
    
    elif config.model_has_xy_indices:
        return GPT2ModelWithXYIndices(model_config)
    
    return CustomGPT2(model_config)


@singledispatch
def get_pretrained_model(c) -> PreTrainedModel:
    """
    Returns already pretrained model
    
    :param c: Specifies path to pretrained model
    :return: Description
    :rtype: PreTrainedModel
    """
    raise Exception(f"Unexpected argument type {type(c)}")


@get_pretrained_model.register(TrainingConfig)
def _1(config: TrainingConfig) -> PreTrainedModel:
    """
    Returns already pretrained model based on training config
    
    :param c: Specifies path to pretrained model
    :type c: TrainingConfig
    :return: Description
    :rtype: PreTrainedModel
    """
    path = config.start_pretrained_model_path
    path = os.path.join(path, "model")

    return get_pretrained_model(path)
        

@get_pretrained_model.register(str)
def _2(path: str) -> PreTrainedModel:
    """
    Returns already pretrained model based on path to saved model on disk
    
    :param c: Specifies path to pretrained model
    :type c: str
    :return: Description
    :rtype: PreTrainedModel
    """
    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as file:
        data = json.load(file)
        
    architecture = data["architectures"][0]

    match architecture:
        case "GPT2ModelWithXYIndices":
            return GPT2ModelWithXYIndices.from_pretrained(path)
        
        case "GPT2ModelWithCornerIndices":
            return GPT2ModelWithVertexIndices.from_pretrained(path)
        
        case "CustomGPT2":
            return CustomGPT2.from_pretrained(path)
        
        case _:
            return AutoModelForCausalLM.from_pretrained(path)


def print_model_size(model: nn.Module):
    """
    Print to stdout number of parameters in a model in millions
    
    :param model: neural network
    :type model: nn.Module
    """

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
