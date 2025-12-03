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

def coord_token(coord: int) -> str:
    return f"<Coord {coord}>"
