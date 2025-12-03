import src.tokens as tokens

import torch


class IgnorePromptInLoss:
    def __init__(self, base_loss):
        self.base_loss = base_loss


    def __call__(self, output, labels):
        # Finding door index
        door_token_mask = (labels == tokens.DOOR_TOKEN_ID)
        door_indices = door_token_mask.int().argmax(dim=1)

        prompt_end_indices = door_indices + 4

        _, cols = labels.shape
        col_idx = torch.arange(cols, device=labels.device).unsqueeze(0)

        prompt_end_indices = prompt_end_indices.unsqueeze(1)

        mask = (col_idx <= prompt_end_indices).to(labels.dtype).bool()
        labels[mask] = -100

        return self.base_loss(output, labels)
