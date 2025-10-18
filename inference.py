import argparse
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import torch
from transformers import AutoModelForCausalLM
from datasets import load_dataset

import floor_plan_tokenizer
from src.drawing import draw_floor_plan
from src.sequence import from_sequence


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

    tokenizer = floor_plan_tokenizer.FloorPlanTokenizer()
    model = AutoModelForCausalLM.from_pretrained(paths_config["trained_model"])

    # valid_dataset = load_dataset(
    #     "text",
    #     data_files=[paths_config["input_data"] + "/validation.txt"]
    # )

    prompt = "<Bound><Coord 95><Coord 64><Coord 95><Coord 52><Coord 119><Coord 52><Coord 119><Coord 68><Coord 185><Coord 68><Coord 185><Coord 148><Coord 165><Coord 148><Coord 165><Coord 195><Coord 109><Coord 195><Coord 109><Coord 205><Coord 72><Coord 205><Coord 72><Coord 64><Door><Coord 75><Coord 64><Coord 91><Coord 64>"
    inputs = tokenizer(prompt, return_tensors="pt",  return_token_type_ids=False).to(model.device)
    model.eval()

    inputs["input_ids"] = inputs["input_ids"][:, 0:-1]
    inputs["attention_mask"] = inputs["attention_mask"][:, 0:-1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

    print("\n=== Sample generation ===\n")
    sample = tokenizer.decode(out[0])
    print(sample)

    floor_plan = from_sequence(sample, "generated")
    draw_floor_plan(floor_plan)


if __name__ == "__main__":
    main()
