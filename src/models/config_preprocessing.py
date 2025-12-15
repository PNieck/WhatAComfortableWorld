from transformers import PreTrainedTokenizer


def preprocess_model_config(config: dict, tokenizer: PreTrainedTokenizer) -> dict:
    if "with_xy_indices" not in config:
        config["with_xy_indices"] = False

    if "with_corner_indices" not in config:
        config["with_corner_indices"] = False

    if "from_checkpoint" not in config:
        config["from_checkpoint"] = False

    config["vocab_size"] = len(tokenizer)

    return config
