import torch
import torch.nn  as nn


class StdLoss:
    def __init__(self):
        self.base_loss_fun = nn.CrossEntropyLoss(ignore_index=-100)


    def __call__(self, output, labels: torch.Tensor):
        logits: torch.Tensor = output.logits

        loss_sum = torch.zeros(1, device=labels.device)

        for i in range(labels.size(0)):
            batch_logits = logits[i,:,:]
            batch_labels = labels[i,:]

            std_loss = self.cross_entropy_loss(batch_logits.unsqueeze(0), batch_labels.unsqueeze(0))

            loss_sum += std_loss

        return loss_sum.squeeze()


    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.base_loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss
