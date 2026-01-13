from regex import I
from src.dataset_loader import load_floor_plans_dataset, Split

from src.sequence import from_sequence
from src.drawing import draw_floor_plan_to_image

from torch.utils.data import DataLoader

dataset = load_floor_plans_dataset("data/gpt2_main/input", Split.VALID)
data_loader = DataLoader(dataset["valid"]["text"], batch_size=1)

i = 0

for seq in data_loader:
    floor_plan = from_sequence(seq[0])
    draw_floor_plan_to_image(floor_plan, f"imgs/validation/{i}.png")
    i += 1

    if i % 100 == 0:
        print(i)
