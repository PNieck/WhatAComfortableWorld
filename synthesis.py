import argparse
import os
import sys

import torch

import src.tokens as tokens

from src.sequence import from_sequence
from src.floor_plan_tokenizer import FloorPlanTokenizer

from src.drawing import draw_floor_plan, draw_floor_plan_to_image
from src.models import (
    print_model_size,
    get_pretrained_model,
)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Generate floor plans from trained model")

    p.add_argument(
        "path_to_model",
        type=str,
        help="Path to the saved model"
    )

    p.add_argument(
        "floor_plans_number",
        type=int,
        help="Number of floor plans to generate"   
    )

    p.add_argument(
        "--draw_imgs",
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
        "--save_seq",
        dest="save_seq",
        type=str,
        help="Output file for generated sequences"
    )

    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    tokenizer = FloorPlanTokenizer()
    model = get_pretrained_model(args.path_to_model)
    print_model_size(model)
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)
    model.eval()

    if args.save_imgs is not None:
        if not os.path.exists(args.save_imgs):
            os.mkdir(args.save_imgs)

    if args.save_seq is not None:
        file = open(args.save_seq, "w")
    else:
        file = None

    for i in range(0, args.floor_plans_number):
        prompt = tokens.BOUNDARY_TOKEN
        
        input = tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False,

        )

        with torch.no_grad():
            # Remove EOS tokens from the end
            for k, v in input.items():
                input[k] = v[:, 0:-1]

            if model.device.type != "cpu":
                input = {k: v.to(model.device) for k, v in input.items()}

            try:
                output = model.generate(
                    **input,
                    max_length = model.max_seq_len,
                    do_sample=True,
                    top_p=0.9,
                    eos_token_id=tokens.END_SEQ_TOKEN_ID,
                    pad_token_id=tokens.PAD_TOKEN_ID,
                    bos_token_id=tokens.START_SEQ_TOKEN_ID,
                    num_beams=1,
                    use_cache=True
                )
            except Exception as e:
                print(f"Exception during generation: {e}")
                continue


        seq = tokenizer.decode(output[0], skip_special_tokens=True)

        if file is not None:
            file.write(seq)
            file.write("\n")

        try:
            floor_plan = from_sequence(seq)
        except Exception as e:
            print(f"Exception during parsing: {e}")
            continue

        if args.save_imgs is not None:
            img_path = os.path.join(args.save_imgs, f"{i}.png")
            draw_floor_plan_to_image(floor_plan, img_path)
            i += 1

        if args.draw_imgs:
            draw_floor_plan(floor_plan)

        print(f"Done {i}/{args.floor_plans_number}")

    if file is not None:
        file.close()



if __name__ == "__main__":
    main(sys.argv[1:])
