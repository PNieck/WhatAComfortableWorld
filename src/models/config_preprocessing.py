from transformers import PreTrainedTokenizer


def preprocess_model_config(config: dict, tokenizer: PreTrainedTokenizer) -> dict:
    if "with_coord_indices" not in config:
        config["with_coord_indices"] = False

    config["vocab_size"] = len(tokenizer)

    return config
