import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np

from src.floor_plan import FloorPlan, RoomType, Room

pastel_colors = [
    "#A6C8E0", "#F4B6A6", "#B8D8BA", "#F2C1C1", "#C6B7E2",
    "#D3B8AE", "#F0CDE3", "#CFCFCF", "#D8E2A8", "#BFE4E8",
    "#E1ECF7", "#FFE0B5", "#CFEBC7"
]

_room2color = {
    RoomType.LivingRoom: pastel_colors[0],
    RoomType.MasterRoom: pastel_colors[1],
    RoomType.Kitchen:    pastel_colors[2],
    RoomType.Bathroom:   pastel_colors[3],
    RoomType.DiningRoom: pastel_colors[4],
    RoomType.ChildRoom:  pastel_colors[5],
    RoomType.StudyRoom:  pastel_colors[6],
    RoomType.SecondRoom: pastel_colors[7],
    RoomType.GuestRoom:  pastel_colors[8],
    RoomType.Balcony:    pastel_colors[9],
    RoomType.Entrance:   pastel_colors[10],
    RoomType.Storage:    pastel_colors[11],
    RoomType.WallIn:     pastel_colors[12]
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


def draw_floor_plan_to_image(floor_plan: FloorPlan, filename: str, with_legend: bool=True):
    fig, ax = plt.subplots()

    ax.axis("off")

    for room in floor_plan.rooms:
        p = Polygon(room.boundary, facecolor=_room2color[room.type], label=room.type.name)

        ax.add_patch(p)

    for room in floor_plan.rooms:
        _draw_room_boundary(ax, room)

    _draw_boundary(ax, floor_plan)
    _draw_doors(ax, floor_plan)

    plt.ylim(0, 250)
    plt.xlim(0, 250)
    plt.gca().invert_yaxis()

    if with_legend:
        plt.legend()

    plt.savefig(filename)

    plt.close()
