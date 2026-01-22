from src.dataset_loader import load_floor_plans_dataset, Split

from src.sequence import from_sequence
from src.drawing import draw_floor_plan_to_image

from torch.utils.data import DataLoader

from src.validation_metrics import BaseMetrics, ErgonomicsTest
from src.floor_plan import RoomType
import math

# dataset = load_floor_plans_dataset("data/gpt2_main/input", Split.VALID)
# data_loader = DataLoader(dataset["valid"]["text"], batch_size=1)

with open("results/seqs/neighbor_loss.txt", "r") as file:
    neighbor_lines = file.readlines()

with open("results/seqs/sota.txt", "r") as file:
    sota_lines = file.readlines()

with open("data/gpt2_main/input/validation.txt", "r") as file:
    valid_lines = file.readlines()

print(f"Neighbor lines: {len(neighbor_lines)}")
print(f"Sota lines: {len(sota_lines)}")
print(f"Valid lines {len(valid_lines)}")

i = 0
base = BaseMetrics()
ergo = ErgonomicsTest()

for neighbor, sota, valid in zip(neighbor_lines, sota_lines, valid_lines):
    # try:
    #     neighbor_floor_plan = from_sequence(neighbor)
    #     sota_floor_plan = from_sequence(sota)
    #     valid_floor_plan = from_sequence(valid)
    # except:
    #     continue

    neighbor_floor_plan = base.filter_out([neighbor])
    sota_floor_plan = base.filter_out([sota])
    valid_floor_plan = base.filter_out([valid])

    if (not neighbor_floor_plan) or (not sota_floor_plan) or (not valid_floor_plan):
        continue

    neighbor_floor_plan = neighbor_floor_plan[0]
    sota_floor_plan = sota_floor_plan[0]
    valid_floor_plan = valid_floor_plan[0]

    neighbor_ergo_loss = ergo.measure_single(neighbor_floor_plan)
    sota_ergo_loss = ergo.measure_single(sota_floor_plan)
    valid_ergo_loss = ergo.measure_single(valid_floor_plan)

    if math.isnan(neighbor_ergo_loss) or math.isnan(sota_ergo_loss) or math.isnan(valid_ergo_loss):
        continue

    if neighbor_ergo_loss < sota_ergo_loss and neighbor_ergo_loss < valid_ergo_loss:
        for room in neighbor_floor_plan.rooms:
            if room.type in {RoomType.Entrance, RoomType.DiningRoom}:
                print(i)

        # draw_floor_plan_to_image(neighbor_floor_plan, f"imgs/paper/best/png/our/our_{i}.png", False)
        # draw_floor_plan_to_image(sota_floor_plan, f"imgs/paper/best/png/baseline/baseline_{i}.png", False)
        # draw_floor_plan_to_image(valid_floor_plan, f"imgs/paper/best/png/gt/gt_{i}.png", False)

        i += 1

        # if i % 100 == 0:
        #     print(i)
