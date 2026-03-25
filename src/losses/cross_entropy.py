import torch.nn  as nn


class CrossEntropyLoss:
    """
    Wrapper class for computing cross-entropy loss with token sequence alignment.
    
    Ignores positions with label value -100.
    """
    
    def __init__(self):
        """
        Initializes the CrossEntropyLoss with default parameters.
        
        Creates a PyTorch CrossEntropyLoss function that ignores tokens with label -100.
        """
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=-100)


    def __call__(self, output, labels):
        """
        Computes the cross-entropy loss between model predictions and labels.
        
        :param output: Model output object containing logits attribute of shape (batch_size, seq_len, vocab_size)
        :param labels: Ground truth token IDs of shape (batch_size, seq_len)
        :return: Scalar cross-entropy loss
        """
        logits = output.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss
