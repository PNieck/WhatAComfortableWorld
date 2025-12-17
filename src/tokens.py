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
    return room_label + CONT_TOKENS_CNT

def room_token(room_label: int) -> str:
    return f"<Room {room_label}>"

def coord_token_id(coord: int):
    return coord + ROOMS_CNT + CONT_TOKENS_CNT

def coord_from_token_id(token_id: int):
    return token_id - ROOMS_CNT - CONT_TOKENS_CNT

def coord_token(coord: int) -> str:
    return f"<Coord {coord}>"


MIN_ROOM_ID = room_token_id(0)
MAX_ROOM_ID = room_token_id(ROOMS_CNT-1)


MIN_COORD_ID = coord_token_id(0)
MAX_COORD_ID = coord_token_id(256)


def is_coord(token_id):
    if isinstance(token_id, int):
        return token_id >= MIN_COORD_ID and token_id <= MAX_COORD_ID
    
    if isinstance(token_id, torch.Tensor):
        return (token_id >= MIN_COORD_ID) & (token_id <= MAX_COORD_ID)
    
    raise Exception(f"Invalid type {type(token_id)}")
