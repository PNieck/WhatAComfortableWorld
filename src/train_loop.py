import torch.nn as nn

from src.training_loops.custom import custom_training_loop
from src.training_loops.transformers import transformers_training_loop


def train(model: nn.Module, tokenizer, dataset, config):
    if config["loop_type"] == "custom":
        custom_training_loop(model, tokenizer, dataset, config)
    elif config["loop_type"] == "transformers":
        transformers_training_loop(model, tokenizer, dataset, config)
