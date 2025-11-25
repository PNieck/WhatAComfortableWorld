import re

from transformers import PreTrainedModel
from datasets import DatasetDict

import torch
from torch.utils.data import DataLoader

import tokens


def _prepare_prompt(seq: str) -> str:
    match = re.search(r"<Room \d+>", seq)
    if not match:
        raise ValueError("No rooms in sequence")

    start = match.start()
    
    return { "text": seq[:start] }


def generate(model: PreTrainedModel, tokenizer, dataset: DatasetDict):
    model.eval()

    prompts = dataset.map(
        lambda ex: _prepare_prompt(ex["text"]),
        batched=False,
    )

    data_loader = DataLoader(prompts["valid"]["text"], batch_size=32)

    generated_sequences = []

    for batch in data_loader:
        inputs = tokenizer(
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

        if model.device.type != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
        with torch.no_grad():

            # Greedy decoding
            outputs = model.generate(
                **inputs,
                max_length = model.max_seq_len,
                do_sample=False,
                eos_token_id=tokens.END_SEQ_TOKEN_ID,
                pad_token_id=tokens.PAD_TOKEN_ID,
                bos_token_id=tokens.START_SEQ_TOKEN_ID,
                use_cache=False,
                num_beams=1
            )

        generated_sequences += tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(f"Generated {len(generated_sequences)} sequences")

    return generated_sequences


