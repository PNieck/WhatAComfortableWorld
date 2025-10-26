from functools import singledispatchmethod
import re

from transformers import PreTrainedTokenizer
import torch

import tokens


class FloorPlanTokenizer(PreTrainedTokenizer):
    def __init__(self, res=256, *args, **kwargs):
        self.resolution = res

        self.unk_token = tokens.UNK_TOKEN
        self.bos_token = tokens.START_SEQ_TOKEN
        self.eos_token = tokens.END_SEQ_TOKEN
        self.pad_token = tokens.PAD_TOKEN

        self.token2id = {
            tokens.UNK_TOKEN       : tokens.UNK_TOKEN_ID,
            tokens.START_SEQ_TOKEN : tokens.START_SEQ_TOKEN_ID,
            tokens.END_SEQ_TOKEN   : tokens.END_SEQ_TOKEN_ID,
            tokens.PAD_TOKEN       : tokens.PAD_TOKEN_ID,
            tokens.BOUNDARY_TOKEN  : tokens.BOUNDARY_TOKEN_ID,
            tokens.DOOR_TOKEN      : tokens.DOOR_TOKEN_ID
        }

        for i in range(tokens.ROOMS_CNT):
            room_id = tokens.room_token_id(i)
            room_token = tokens.room_token(i)
            
            self.token2id[room_token] = room_id

        for i in range(self.resolution):
            coord_id = tokens.coord_token_id(i)
            coord_token = tokens.coord_token(i)

            self.token2id[coord_token] = coord_id

        self.id2token = dict((v, k) for k, v in self.token2id.items())

        super().__init__(
            unk_token=self.unk_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            *args, **kwargs
        )


    def __len__(self):
        return len(self.token2id)


    def _tokenize(self, text):
        return re.findall(r'[^>]+>', text)


    def _convert_token_to_id(self, token):
        return self.token2id[token]


    def _convert_id_to_token(self, index):
        return self.id2token[index]


    def convert_tokens_to_string(self, tokens):
        """Join characters back into text."""
        return "".join(tokens)


    def build_inputs_with_special_tokens(self, token_ids, _):
        """Wrap input in start and end sequence tokens."""
        return [tokens.START_SEQ_TOKEN_ID] + token_ids + [tokens.END_SEQ_TOKEN_ID]


    def get_vocab(self):
        return self.token2id

    @property
    def vocab_size(self) -> int:
        return len(self.token2id) - 4


    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        # For GPT-style models, we don’t use sentence pairs.
        # So we just return a list of 0s with the same length as input_ids.
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)  # +2 for <s> and </s>
        else:
            # If you ever add sentence-pair mode
            return [0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 1)
    
    @singledispatchmethod
    def is_coord_token(self, token):
        raise TypeError(f"Invalid type: {type(token)}")
        
    @is_coord_token.register
    def _(self, token: torch.Tensor) -> torch.Tensor:
        min_coord_token = tokens.coord_token_id(0)
        max_coord_token = tokens.coord_token_id(self.resolution)

        return (token >= min_coord_token) & (token <= max_coord_token)
    
    @is_coord_token.register
    def _(self, token: list):
        return [self.is_coord_token(val) for val in token]
    
    @is_coord_token.register
    def _(self, token: int) -> bool:
        min_coord_token = tokens.coord_token_id(0)
        max_coord_token = tokens.coord_token_id(self.resolution)

        return token >= min_coord_token and token <= max_coord_token
