from transformers import GPT2Config, GPT2LMHeadModel

import tokens


def get_gpt2_config(config) -> GPT2Config:
    return GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config["max_seq_len"],
        n_ctx=config["max_seq_len"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        bos_token_id=tokens.START_SEQ_TOKEN_ID,
        eos_token_id=tokens.END_SEQ_TOKEN_ID,
    )


def get_gpt2(config):
    config = get_gpt2_config(config)

    model = GPT2LMHeadModel(config)

    return model
