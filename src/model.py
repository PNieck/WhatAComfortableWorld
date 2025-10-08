import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
       

class FloorPlanGenModel(nn.Module):
    

    def __init__(self, config, tokens_cnt):
        super(FloorPlanGenModel, self).__init__()

        gemmaConfig = AutoConfig.from_pretrained("google/gemma-3-270m")

        if "hidden_size" in config:
            gemmaConfig.hidden_size = config["hidden_size"]

        if "max_seq_len" in config:
            gemmaConfig.max_position_embeddings = config["max_seq_len"]

        if "sliding_window" in config:
            gemmaConfig.sliding_window = config["sliding_window"]

        if "intermediate_size" in config:
            gemmaConfig.intermediate_size = config["intermediate_size"]

        self.model = AutoModelForCausalLM.from_config(gemmaConfig, attn_implementation='eager')

        self.model.resize_token_embeddings(tokens_cnt)


    def forward(
        self,
        input_ids=None,
        labels=None,
        token_type_ids=None,
        attention_mask=None
    ):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
    

    def save(self, path):
        self.model.save_pretrained(path)
