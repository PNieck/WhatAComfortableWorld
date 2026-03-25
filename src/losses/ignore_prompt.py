"""Loss wrapper for ignoring prompt tokens in loss computation.

Provides wrapper class that masks out prompt section of sequences from loss
calculation to focus training on room and geometry generation.
"""

import src.tokens as tokens

import torch


class IgnorePromptInLoss:
    """Wrapper class that masks out prompt tokens from loss computation."""
    
    def __init__(self, base_loss):
        """Initialize IgnorePromptInLoss wrapper.
        
        :param base_loss: The base loss function to wrap
        """
        self.base_loss = base_loss


    def __call__(self, output, labels):
        """Compute loss while ignoring prompt tokens.
        
        Identifies the prompt section (up to the first door token) and masks
        those positions to -100 so they are ignored by the loss function.
        
        :param output: Model output containing logits
        :param labels: Ground truth token IDs of shape (batch_size, seq_len)
        :return: Loss value computed by base loss function on non-prompt tokens
        """

        # Finding door index
        door_token_mask = (labels == tokens.DOOR_TOKEN_ID)
        door_indices = door_token_mask.int().argmax(dim=1)

        prompt_end_indices = door_indices + 4

        cols = labels.shape[1]
        col_idx = torch.arange(cols, device=labels.device).unsqueeze(0)

        prompt_end_indices = prompt_end_indices.unsqueeze(1)

        mask = (col_idx <= prompt_end_indices).to(labels.dtype).bool()
        labels[mask] = -100

        return self.base_loss(output, labels)
