import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np

from src.floor_plan import FloorPlan, RoomType, Room


_room2color = {
    RoomType.LivingRoom: "red",
    RoomType.MasterRoom: "green",
    RoomType.Kitchen: "blue",
    RoomType.Bathroom: "orange",
    RoomType.DiningRoom: "violet",
    RoomType.ChildRoom: "olive",
    RoomType.StudyRoom: "aqua",
    RoomType.SecondRoom: "pink",
    RoomType.GuestRoom: "silver",
    RoomType.Balcony: "magenta",
    RoomType.Entrance: "gold",
    RoomType.Storage: "navy",
    RoomType.WallIn: "brown"
}


def _draw_doors(ax, floor_plan: FloorPlan):
    x = floor_plan.front_door.xs
    y = floor_plan.front_door.ys

    ax.plot(x, y, linewidth=2, color="yellow")


def _draw_room_boundary(ax, room: Room):
    x = room.boundary[:, 0]
    y = room.boundary[:, 1]

    x = np.append(x, room.boundary[0, 0])
    y = np.append(y, room.boundary[0, 1])

    ax.plot(x, y, linewidth=2, color="grey")


def _draw_boundary(ax, floor_plan: FloorPlan):
    x = floor_plan.boundary[:, 0]
    y = floor_plan.boundary[:, 1]

    x = np.append(x, floor_plan.boundary[0, 0])
    y = np.append(y, floor_plan.boundary[0, 1])

    ax.plot(x, y, linewidth=2, color="black")


def _draw_room_corners(ax, floor_plan: FloorPlan):
    for room in floor_plan.rooms:
        x = room.boundary[:, 0]
        y = room.boundary[:, 1]

        ax.scatter(x, y, color="black", s=10)



def draw_floor_plan(floor_plan: FloorPlan, draw_room_corners: bool = False):
    fig, ax = plt.subplots()

    for room in floor_plan.rooms:
        p = Polygon(room.boundary, facecolor=_room2color[room.type], label=room.type.name)

        ax.add_patch(p)

    for room in floor_plan.rooms:
        _draw_room_boundary(ax, room)

    _draw_boundary(ax, floor_plan)
    _draw_doors(ax, floor_plan)

    if draw_room_corners:
        _draw_room_corners(ax, floor_plan)

    plt.grid()
    plt.ylim(0, 250)
    plt.xlim(0, 250)
    plt.gca().invert_yaxis()
    plt.legend()

    plt.title(f'Floor plan {floor_plan.name}')

    plt.show()
