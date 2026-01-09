import argparse
import os
import json

import torch

from src.floor_plan_tokenizer import FloorPlanTokenizer
from src.generation import Generator
from src.drawing import draw_floor_plan, draw_floor_plan_to_image
from src.dataset_loader import load_floor_plans_dataset, Split
from src.models import (
    print_model_size,
    get_pretrained_model,
)

from src.validation_metrics import (
    ParsabilityRate,
    CoverageTest,
    GeomValidityRate,
    RoomsOverlappingTest,
    RequiredRoomsTest,
    NarrowSpacesTest,
    GeometrySimplicityTest,
    RoomsNeighborhoodTest,
    BaseMetrics,
    ErgonomicsTest
)


def main():
    p = argparse.ArgumentParser(description="Generate floor plans from trained model")

    p.add_argument(
        "path_to_model",
        type=str,
        help="Path to the saved model"
    )

    p.add_argument(
        "path_to_dataset",
        type=str,
        help="Path to preprocessed data from RPLAN dataset"
    )

    p.add_argument(
        "--masked",
        action="store_true",
        help="Use masked inference. If provided, overrides 'use_masked_inference' from config."
    )

    p.add_argument(
        "--draw_images",
        action="store_true",
        help="If specified, generated images are displayed on a screen"
    )

    p.add_argument(
        "--save_imgs",
        dest="save_imgs",
        type=str,
        help="Output directory for generated floor plan images"
    )

    p.add_argument(
        "--save_output",
        dest="save_output",
        type=str,
        help="Output directory metrics"
    )

    args = p.parse_args()

    tokenizer = FloorPlanTokenizer()
    model = get_pretrained_model(args.path_to_model)
    print_model_size(model)
    print(model)

    if args.masked:
        model.use_masked_inference = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)
    model.eval()

    dataset = load_floor_plans_dataset(args.path_to_dataset, Split.VALID)

    pars_rate = ParsabilityRate()
    validity_rate = GeomValidityRate()
    geom_simplicity = GeometrySimplicityTest()
    cov_rate = CoverageTest()
    room_overlap_rate = RoomsOverlappingTest()
    required_rooms = RequiredRoomsTest()
    narrow_spaces = NarrowSpacesTest()
    neighborhood = RoomsNeighborhoodTest()
    base_metrics = BaseMetrics()
    ergonomics = ErgonomicsTest()

    # TODO: set bigger batch size
    batch = 32
    if args.masked:
        batch = 1

    generator = Generator(model, tokenizer, dataset, batch)

    done = 0
    total = len(dataset["valid"])

    if args.save_imgs is not None:
        if not os.path.exists(args.save_imgs):
            os.mkdir(args.save_imgs)

    i = 0
    for batch in generator.generate_in_batches():
        floor_plans = pars_rate.parse(batch)
        floor_plans = validity_rate.filter_out_invalid(floor_plans)
        geom_simplicity.simplify(floor_plans)
        cov_rate.measure(floor_plans)
        room_overlap_rate.measure(floor_plans)
        required_rooms.measure(floor_plans)
        narrow_spaces.measure(floor_plans)
        neighborhood.measure(floor_plans)
        base_metrics.measure(batch)
        ergonomics.measure(floor_plans)

        if args.save_imgs is not None:
            for plan in floor_plans:
                img_path = os.path.join(args.save_imgs, f"{i}.png")
                draw_floor_plan_to_image(plan, img_path)
                i += 1

        if args.draw_images:
            for plan in floor_plans:
                draw_floor_plan(plan)

        done += len(batch)
        print(f"Done {done}/{total}")

    print(f"Generations fails: {generator.fails}")
    print(f"Validity problems {model.validity_problems}")

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
    print(f"Fully covered floor plans: {cov_rate.correct_floor_plans}/{cov_rate.examples_cnt} ({cov_rate.correctness_rate()}%)")

    print("\n")
    print(f"Rooms avg overlapping rate: {room_overlap_rate.avg_overlapping_rate()}")
    print(f"Floor plans with no overlapping rooms: {room_overlap_rate.correct_floor_plans}/{room_overlap_rate.examples_cnt} ({room_overlap_rate.correctness_rate()}%)")

    print("\n")
    base_metrics.print_results()

    print("\n")
    print(f"Required rooms rate: {required_rooms.correctness_rate()}")
    print(f"Floor plans with all required rooms: {required_rooms.correct_floor_plans}/{required_rooms.examples_cnt} ({required_rooms.correctness_rate()}%)")
    required_rooms.print_missing_rooms()

    print("\n")
    print(f"Floor plans with no narrow spaces: {narrow_spaces.correct_cnt}/{narrow_spaces.examples_cnt} ({narrow_spaces.correctness_rate()}%)")

    print("\n")
    print(f"Avg neighborhood loss: {neighborhood.avg_loss()}")
    print(f"Perfect neighborhood floor plans: {neighborhood.perfect_floor_plans}/{neighborhood.examples_cnt} ({neighborhood.correctness_rate()}%)")
    if neighborhood.nan_losses > 0:
        print(f"Nan neighbor losses: {neighborhood.nan_losses}")

    print("\n")
    print(f"Avg ergonomics loss: {ergonomics.avg_loss()}")
    print(f"Perfect ergonomics floor plans: {ergonomics.perfect_floor_plans}/{ergonomics.examples_cnt} ({ergonomics.correctness_rate()}%)")
    if ergonomics.nan_losses > 0:
        print(f"Nan neighbor losses: {ergonomics.nan_losses}")

    if args.save_output is not None:
        output = {}

        output["Generations fails"] = generator.fails
        output["Validity generation problems"] = model.validity_problems

        output["Parsability"] = {
            "rate": pars_rate.rate(),
            "examples": pars_rate.examples_cnt,
            "failures": pars_rate.invalid_seq,
            "error_types": pars_rate.error_types
        }

        output["Validity"] = {
            "rate": validity_rate.rate(),
            "valid examples": validity_rate.valid_examples
        }

        output["Simplicity rate"] = geom_simplicity.rate()

        output["Coverage rate"] = cov_rate.correctness_rate()

        output["Overlapping rate"] = room_overlap_rate.correctness_rate()

        output["Base"] = base_metrics.success_rate()

        output["Required rooms"] = required_rooms.correctness_rate()

        output["Narrow spaces"] = narrow_spaces.correctness_rate()

        output["Ergonomic loss"] = {
            "avg loss": ergonomics.avg_loss(),
            "rate": ergonomics.correctness_rate()
        }

        with open(args.save_output, 'w') as f:
            json.dump(output, f, indent=4)



if __name__ == "__main__":
    main()
