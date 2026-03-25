"""Generation module for floor plan sequence generation.

Provides a Generator class for performing batch-based inference with a pre-trained
model using greedy decoding on floor plan generation prompts.
"""

import re

from transformers import PreTrainedModel
from datasets import DatasetDict

import torch
from torch.utils.data import DataLoader

import src.tokens as tokens


class Generator:
    """Generator for creating floor plan sequences from prompts.
    
    Performs batch-based inference on prompts extracted from a dataset, using greedy
    decoding to generate complete floor plan sequences. Handles tokenization, device
    placement, and error handling during generation.
    """

    def __init__(self, model: PreTrainedModel, tokenizer, dataset: DatasetDict, batch_size: int =32):
        """Initialize the Generator.
        
        :param model: Pre-trained language model for sequence generation
        :param tokenizer: Tokenizer for encoding prompts and decoding outputs
        :param dataset: Dataset containing sequences with prompts to generate from
        :param batch_size: Number of sequences to process per batch (default: 32)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size

        self.fails = 0

    def _prepare_prompt(self, seq: str):
        """Extract the prompt portion of a sequence up to the first room token.
        
        :param seq: Full sequence containing room tokens and coordinates
        :return: Dictionary with "text" key containing the prompt portion
        :raises ValueError: If no room tokens are found in the sequence
        """
        match = re.search(r"<Room \d+>", seq)
        if not match:
            raise ValueError("No rooms in sequence")

        start = match.start()
        
        return { "text": seq[:start] }


    def generate_in_batches(self):
        """Generate floor plan sequences in batches using greedy decoding.
        
        Extracts prompts from the dataset, processes them in batches through the model,
        and yields generated sequences. Sets model to evaluation mode and handles device
        placement. Failed generations are tracked but do not interrupt the process.
        
        :return: Generator yielding lists of decoded sequences for each batch
        """
        self.model.eval()

        prompts = self.dataset.map(
            lambda ex: self._prepare_prompt(ex["text"]),
            batched=False,
        )

        data_loader = DataLoader(prompts["test"]["text"], batch_size=self.batch_size)

        for batch in data_loader:
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
                padding_side="left",
                return_token_type_ids=False
            )

            # Remove EOS tokens from the end
            for k, v in inputs.items():
                inputs[k] = v[:, 0:-1]

            if self.model.device.type != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
            with torch.no_grad():

                # Greedy decoding
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_length = self.model.max_seq_len,
                        do_sample=False,
                        eos_token_id=tokens.END_SEQ_TOKEN_ID,
                        pad_token_id=tokens.PAD_TOKEN_ID,
                        bos_token_id=tokens.START_SEQ_TOKEN_ID,
                        num_beams=1,
                        use_cache=True
                    )
                except Exception as e:
                    print(f"Exception during generation: {e}")
                    self.fails += 1
                    continue

            generated_sequences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            yield generated_sequences
