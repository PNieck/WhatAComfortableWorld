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

from src.validation_metrics import (
    ParsabilityRate,
    CoverageTest,
    GeomValidityRate,
    RoomsOverlappingTest,
    RequiredRoomsTest,
    NarrowSpacesTest,
    GeometrySimplicityTest,
    RoomsNeighborhoodTest
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
    geom_simplicity = GeometrySimplicityTest()
    cov_rate = CoverageTest()
    room_overlap_rate = RoomsOverlappingTest()
    required_rooms = RequiredRoomsTest()
    narrow_spaces = NarrowSpacesTest()
    neighborhood = RoomsNeighborhoodTest()

    generator = Generator(model, tokenizer, dataset)

    done = 0
    total = len(dataset["valid"])

    for batch in generator.generate_in_batches():
        floor_plans = pars_rate.parse(batch)
        floor_plans = validity_rate.filter_out_invalid(floor_plans)
        geom_simplicity.simplify(floor_plans)
        cov_rate.measure(floor_plans)
        room_overlap_rate.measure(floor_plans)
        required_rooms.measure(floor_plans)
        narrow_spaces.measure(floor_plans)
        neighborhood.measure(floor_plans)

        done += len(batch)
        print(f"Done {done}/{total}")

    print(f"Parsability: {pars_rate.rate()}")
    print(f"Examples {pars_rate.examples_cnt}")
    print(f"Failures {pars_rate.invalid_seq}")

    print("\n")
    print(pars_rate.error_types)

    print("\n")
    print(f"Validity rate {validity_rate.rate()}")
    print(f"Valid examples {validity_rate.valid_examples}")

    print("\n")
    print(f"Simplicity rate {geom_simplicity.rate()}")

    print("\n")
    print(f"Room coverage: {cov_rate.avg_coverage_rate()}")
    print(f"Overfill rate {cov_rate.avg_overfilling_rate()}")
    print(f"Fully covered floor plans: {cov_rate.correct_floor_plans}/{cov_rate.examples_cnt}")

    print("\n")
    print(f"Rooms avg overlapping rate: {room_overlap_rate.avg_overlapping_rate()}")
    print(f"Floor plans with no overlapping rooms: {room_overlap_rate.correct_floor_plans}/{room_overlap_rate.examples_cnt}")

    print("\n")
    print(f"Rooms avg overlapping rate: {room_overlap_rate.avg_overlapping_rate()}")
    print(f"Floor plans with no overlapping rooms: {room_overlap_rate.correct_floor_plans}/{room_overlap_rate.examples_cnt}")

    print("\n")
    print(f"Required rooms rate: {required_rooms.correctness_rate()}")
    print(f"Floor plans with all required rooms: {required_rooms.correct_floor_plans}/{required_rooms.examples_cnt}")
    required_rooms.print_missing_rooms()

    print("\n")
    print(f"Floor plans with no narrow spaces: {narrow_spaces.correct_cnt}/{narrow_spaces.examples_cnt}")

    print("\n")
    print(f"Avg neighborhood loss: {neighborhood.avg_loss()}")
    print(f"Perfect neighborhood floor plans: {neighborhood.perfect_floor_plans}/{neighborhood.examples_cnt}")
    if neighborhood.nan_losses > 0:
        print(f"Nan neighbor losses: {neighborhood.nan_losses}")

if __name__ == "__main__":
    main()
