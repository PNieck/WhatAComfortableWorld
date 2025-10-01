import torch.nn as nn

from transformers import AutoConfig, AutoModel


class FloorPlanGenModel(nn.Module):
    def __init__(self, tokens_cnt):
        super(FloorPlanGenModel, self).__init__()

        gemmaConfig = AutoConfig.from_pretrained("google/gemma-3-270m")
        self.model = AutoModel.from_config(gemmaConfig)

        self.model.resize_token_embeddings(tokens_cnt)

    def forward(self, data):
        return self.model(data)
