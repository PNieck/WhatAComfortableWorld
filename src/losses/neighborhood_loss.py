import torch
import torch.nn.functional as F


class NeighborhoodLoss:
    def __init__(self, base_loss):
        self.base_loss = base_loss

    def __call__(self, output, labels: torch.Tensor):
        std_loss = self.base_loss(output, labels)
        logits = output.logits

        probs = F.softmax(logits, -1)

    # def evaluate(self, seq):
        
