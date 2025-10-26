from transformers import AutoConfig, AutoModelForCausalLM


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
