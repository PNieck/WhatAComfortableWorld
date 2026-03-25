"""Token definitions and utilities for floor plan sequence tokenization."""

from typing import overload

import torch


START_SEQ_TOKEN_ID = 0
START_SEQ_TOKEN = "<BOS>"

END_SEQ_TOKEN_ID = 1
END_SEQ_TOKEN = "<EOS>"

UNK_TOKEN_ID = 2
UNK_TOKEN = "<UNK>"

PAD_TOKEN_ID = 3
PAD_TOKEN = "<PAD>"

BOUNDARY_TOKEN_ID = 4
BOUNDARY_TOKEN = "<Bound>"

DOOR_TOKEN_ID = 5
DOOR_TOKEN = "<Door>"

ROOMS_CNT = 13
CONT_TOKENS_CNT = 6


def room_token_id(room_label: int):
    """Convert room label to its token ID.
    
    :param room_label: Room label (0-12)
    :return: Token ID for the room
    """
    return room_label + CONT_TOKENS_CNT

def room_token(room_label: int) -> str:
    """Create string representation of a room token.
    
    :param room_label: Room label (0-12)
    :return: String representation like "<Room 0>"
    """
    return f"<Room {room_label}>"

def coord_token_id(coord: int):
    """Convert coordinate value to its token ID.
    
    :param coord: Coordinate value (0-255)
    :return: Token ID for the coordinate
    """
    return coord + ROOMS_CNT + CONT_TOKENS_CNT

def coord_from_token_id(token_id: int | torch.Tensor):
    """Extract coordinate value from its token ID.
    
    :param token_id: Token ID for a coordinate
    :return: Coordinate value (0-255)
    """
    return token_id - ROOMS_CNT - CONT_TOKENS_CNT

def coord_token(coord: int) -> str:
    """Create string representation of a coordinate token.
    
    :param coord: Coordinate value (0-255)
    :return: String representation like "<Coord 0>"
    """
    return f"<Coord {coord}>"


MIN_ROOM_ID = room_token_id(0)
MAX_ROOM_ID = room_token_id(ROOMS_CNT-1)


MIN_COORD_ID = coord_token_id(0)
MAX_COORD_ID = coord_token_id(256)


@overload
def is_coord(token_id: int) -> bool:
    ...

@overload
def is_coord(token_id: torch.Tensor) -> torch.Tensor:
    ...

def is_coord(token_id: int|torch.Tensor) -> bool | torch.Tensor:
    """Check if a token ID represents a coordinate token.
    
    :param token_id: Token ID or tensor of token IDs
    :return: Boolean or tensor of booleans indicating if token is a coordinate
    :raises Exception: If token_id is not int or torch.Tensor
    """
    if isinstance(token_id, int):
        return token_id >= MIN_COORD_ID and token_id <= MAX_COORD_ID
    
    if isinstance(token_id, torch.Tensor):
        return (token_id >= MIN_COORD_ID) & (token_id <= MAX_COORD_ID)
    
    raise Exception(f"Invalid type {type(token_id)}")


def is_room(token_id) -> bool | torch.Tensor:
    """Check if a token ID represents a room token.
    
    :param token_id: Token ID or tensor of token IDs
    :return: Boolean or tensor of booleans indicating if token is a room
    :raises Exception: If token_id is not int or torch.Tensor
    """
    if isinstance(token_id, int):
        return token_id >= MIN_ROOM_ID and token_id <= MAX_ROOM_ID
    
    if isinstance(token_id, torch.Tensor):
        return (token_id >= MIN_ROOM_ID) & (token_id <= MAX_ROOM_ID)
    
    raise Exception(f"Invalid type {type(token_id)}")
