import re

from transformers import PreTrainedModel
from datasets import DatasetDict

import torch
from torch.utils.data import DataLoader

import src.tokens as tokens


class Generator:
    def __init__(self, model: PreTrainedModel, tokenizer, dataset: DatasetDict, batch_size: int =32):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size

        self.fails = 0

    def _prepare_prompt(self, seq: str):
        match = re.search(r"<Room \d+>", seq)
        if not match:
            raise ValueError("No rooms in sequence")

        start = match.start()
        
        return { "text": seq[:start] }


    def generate_in_batches(self):
        self.model.eval()

        prompts = self.dataset.map(
            lambda ex: self._prepare_prompt(ex["text"]),
            batched=False,
        )

        data_loader = DataLoader(prompts["valid"]["text"], batch_size=self.batch_size)

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
                except:
                    self.fails += 1

            generated_sequences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            yield generated_sequences
