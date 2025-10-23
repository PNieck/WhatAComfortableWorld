from transformers import (
    AutoConfig, AutoModelForCausalLM,
    GPT2Config, GPT2LMHeadModel,
)

import torch.nn as nn


def get_gemma3(config, tokens_cnt):
    gemmaConfig = AutoConfig.from_pretrained("google/gemma-3-270m")

    if "hidden_size" in config:
        gemmaConfig.hidden_size = config["hidden_size"]

    if "max_seq_len" in config:
        gemmaConfig.max_position_embeddings = config["max_seq_len"]

    if "sliding_window" in config:
        gemmaConfig.sliding_window = config["sliding_window"]

    if "intermediate_size" in config:
        gemmaConfig.intermediate_size = config["intermediate_size"]

    model = AutoModelForCausalLM.from_config(gemmaConfig, attn_implementation='eager')
    model.resize_token_embeddings(tokens_cnt)

    model.apply(model._init_weights)

    return model


def get_gpt2(config, tokens_cnt):
    config = GPT2Config(
        vocab_size=tokens_cnt,
        n_positions=config["max_seq_len"],
        n_ctx=config["max_seq_len"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        bos_token_id=0,  # will be set by tokenizer when resized
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)

    model.resize_token_embeddings(tokens_cnt)

    return model


def get_model(config, tokens_cnt) -> nn.Module:
    if config["type"] == "gemma3":
        return get_gemma3(config, tokens_cnt)
    elif config["type"] == "gpt2":
        return get_gpt2(config, tokens_cnt)
    elif config["type"] == "existing":
        return AutoModelForCausalLM.from_pretrained(config["input_model_path"])
    else:
        raise ValueError("Invalid model type")
    

def print_model(model: nn.Module):
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
