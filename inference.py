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


def draw_floor_plan(tokens):
    tokenizer = floor_plan_tokenizer.FloorPlanTokenizer()
    tokenized = tokenizer.tokenize(tokens)

    print(tokenized)


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

    valid_dataset = load_dataset(
        "text",
        data_files=[paths_config["input_data"] + "/validation.txt"]
    )

    prompt = "<Bound><Coord 193><Coord 103><Coord 197><Coord 103><Coord 197><Coord 208><Coord 153><Coord 208><Coord 153><Coord 180><Coord 60><Coord 180><Coord 60><Coord 75><Coord 68><Coord 75><Coord 68><Coord 78><Coord 134><Coord 78><Coord 134><Coord 51><Coord 197><Coord 51><Coord 197><Coord 73><Coord 193><Coord 73>"
    inputs = tokenizer(prompt, return_tensors="pt",  return_token_type_ids=False).to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

    print("\n=== Sample generation ===\n")
    print(tokenizer.decode(out[0]))

    print("\n\n")
    print(out[0])


if __name__ == "__main__":
    main()
