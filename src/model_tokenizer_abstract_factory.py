from typing import Tuple

from src.tokenizers import FloorPlanTokenizer
from src.tokenizers import FloorPlanWithCoordIndicesTokenizer

from src.models import get_model
from src.models.gpt2_with_coord_indices import GPT2ModelWithCoordIndices

import torch.nn as nn
from transformers import AutoModelForCausalLM


def get_model_and_tokenizer(config) -> Tuple[nn.Module, FloorPlanTokenizer]:
    if "with_coord_indices" not in config:
        config["with_coord_indices"] = False

    if config["with_coord_indices"]:
        tokenizer = FloorPlanWithCoordIndicesTokenizer()
    else:
        tokenizer = FloorPlanTokenizer()

    model = get_model(config, len(tokenizer))

    return model, tokenizer


def get_pretrained_model_and_tokenizer(path, with_coord_indices: bool) -> Tuple[nn.Module, FloorPlanTokenizer]:
    if with_coord_indices:
        model = GPT2ModelWithCoordIndices.from_pretrained(path)
    else:
        model = AutoModelForCausalLM.from_pretrained(path)

    if isinstance(model, GPT2ModelWithCoordIndices):
        tokenizer = FloorPlanWithCoordIndicesTokenizer()
    else:
        tokenizer = FloorPlanTokenizer()
    
    return model, tokenizer
