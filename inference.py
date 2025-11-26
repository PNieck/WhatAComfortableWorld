import argparse
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import torch

from src.floor_plan_tokenizer import FloorPlanTokenizer
from src.generation import Generator
from src.drawing import draw_floor_plan
from src.dataset_loader import load_floor_plans_dataset, Split
from src.models import (
    print_model_size,
    get_pretrained_model,
    preprocess_model_config
)

from src.inference_metrics import (
    ParsabilityRate,
    CoverageTest,
    GeomValidityRate
)


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
    model_config = config["model"]

    tokenizer = FloorPlanTokenizer()
    model_config = preprocess_model_config(model_config, tokenizer)
    model = get_pretrained_model(paths_config["trained_model"])
    print_model_size(model)
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)
    model.eval()

    dataset = load_floor_plans_dataset(paths_config["input_data"], Split.VALID)

    pars_rate = ParsabilityRate()
    validity_rate = GeomValidityRate()
    cov_rate = CoverageTest()

    generator = Generator(model, tokenizer, dataset)

    for batch in generator.generate_in_batches():
        floor_plans = pars_rate.parse(batch)
        floor_plans = validity_rate.filter_out_invalid(floor_plans)
        cov_rate.measure(floor_plans)

        print("Batch done")

    print(f"Parsability: {pars_rate.rate()}")
    print(f"Examples {pars_rate.examples_cnt}")
    print(f"Failures {pars_rate.invalid_seq}")

    print("\n")
    print(pars_rate.error_types)

    print("\n")
    print(f"Validity rate {validity_rate.rate()}")
    print(f"Valid examples {validity_rate.valid_examples}")

    print("\n")
    print(f"Room coverage: {cov_rate.avg_coverage_rate()}")
    print(f"Overfill rate {cov_rate.avg_overfilling_rate()}")
    print(f"Fully covered floor plans: {cov_rate.correct_floor_plans}")


if __name__ == "__main__":
    main()
