import torch.nn  as nn


class CrossEntropyLoss:
    def __init__(self):
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=-100)

    def __call__(self, output, labels):
        logits = output.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss
