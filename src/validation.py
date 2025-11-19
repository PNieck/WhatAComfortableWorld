import re

from transformers import PreTrainedModel

from datasets import DatasetDict

import torch
from torch.utils.data import DataLoader

from src.log_writer import LogWriter
from src.inference_metrics import (
    ParsabilityRate,
    CoverageTest,
    GeomValidityRate
)

import tokens


def prepare_prompt(seq: str) -> str:
    match = re.search(r"<Room \d+>", seq)
    if not match:
        raise ValueError("No rooms in sequence")

    start = match.start()
    
    return { "text": seq[:start] }


def validate(
        model: PreTrainedModel,
        tokenizer, dataset: DatasetDict,
        training_config: dict,
        model_config: dict,
        log_writer: LogWriter
    ):
    model.eval()

    prompts = dataset.map(
        lambda ex: prepare_prompt(ex["text"]),
        batched=False,
    )
    
    data_loader = DataLoader(prompts["valid"]["text"], batch_size=32)

    pars_rate = ParsabilityRate()
    validity_rate = GeomValidityRate()
    cov_rate = CoverageTest()

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
                max_length = model_config["max_seq_len"],
                do_sample=False,
                eos_token_id=tokens.END_SEQ_TOKEN_ID,
                pad_token_id=tokens.PAD_TOKEN_ID,
                bos_token_id=tokens.START_SEQ_TOKEN_ID,
                use_cache=False,
                num_beams=1
            )
        
        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        floor_plans = pars_rate.parse(results)
        if not floor_plans:
            continue

        floor_plans = validity_rate.filter_out_invalid(floor_plans)
        if not floor_plans:
            continue

        cov_rate.measure(floor_plans)

    hparams = training_config.copy()
    hparams = hparams | model_config

    hparams.pop("eval_steps", None)
    hparams.pop("log_comment", None)
    hparams.pop("name", None)

    metrics = {}

    pars_rate.add_to_metrics(metrics)
    validity_rate.add_to_metrics(metrics)
    cov_rate.add_to_metrics(metrics)

    log_writer.add_hparams(hparams, metrics)
