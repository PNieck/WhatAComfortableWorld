import argparse
import re
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM

import src.tokenizers.floor_plan_tokenizer as floor_plan_tokenizer
from src.drawing import draw_floor_plan
from src.dataset_loader import load_floor_plans_dataset, Split
from src.models import print_model
from src.model_tokenizer_abstract_factory import get_pretrained_model_and_tokenizer
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


def main():
    p = argparse.ArgumentParser(description="Generate floor plans from trained model")

    p.add_argument(
        "path_to_config",
        type=str,
        help="Path to the configuration file"
    )
    args = p.parse_args()

    with open(args.path_to_config, "r") as f:
        config = yaml.load(f, Loader=Loader)

    paths_config = config["paths"]

    b = "with_coord_indices" in config["model"]
    model, tokenizer = get_pretrained_model_and_tokenizer(paths_config["trained_model"], b)
    print_model(model)
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = load_floor_plans_dataset(paths_config["input_data"], Split.VALID)
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

        if device.type != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Remove EOS tokens from the end
        for k, v in inputs.items():
            inputs[k] = v[:, 0:-1]

        with torch.no_grad():

            # Greedy decoding
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                eos_token_id=tokens.END_SEQ_TOKEN_ID
            )

        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        floor_plans = pars_rate.parse(results)
        if not floor_plans:
            continue

        floor_plans = validity_rate.filter_out_invalid(floor_plans)
        if not floor_plans:
            continue

        # for plan in floor_plans:
        #     draw_floor_plan(plan)

        cov_rate.measure(floor_plans)

    print(f"Parsability: {pars_rate.rate()}")
    print(f"Examples {pars_rate.examples_cnt}")
    print(f"Failures {pars_rate.invalid_seq}")

    print("\n")
    print(pars_rate.error_types)

    print("\n")
    print(f"Validity rate {validity_rate.rate()}")
    print(f"Valid examples {validity_rate.valid_examples}")

    print("\n")
    print(f"Room coverage: {cov_rate.coverage_rate()}")
    print(f"Outside area rate {cov_rate.area_outside_rate()}")
    print(f"Fully covered floor plans: {cov_rate.correct_floor_plans}")


if __name__ == "__main__":
    main()
