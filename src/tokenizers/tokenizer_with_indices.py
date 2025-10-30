from functools import singledispatchmethod
from typing import Union

from .floor_plan_tokenizer import FloorPlanTokenizer
import torch


class FloorPlanWithCoordIndicesTokenizer(FloorPlanTokenizer):
    def __init__(self, res=256, *args, **kwargs):
        super().__init__(res, *args, **kwargs)

    def __call__(
            self,
            text: Union[str, list[str], list[list[str]]] = None,    # The sequence or batch of sequences to be encoded
            text_pair = None,
            text_target = None,
            text_pair_target = None,
            add_special_tokens = True,
            padding = False,
            truncation = None,
            max_length = None,
            stride = 0,
            is_split_into_words = False,    # Whether or not the input is already pre-tokenized
            pad_to_multiple_of = None,
            padding_side = None,
            return_tensors = None,
            return_token_type_ids = None,
            return_attention_mask = None,
            return_overflowing_tokens = False,
            return_special_tokens_mask = False,
            return_offsets_mapping = False, # Whether or not to return (char_start, char_end) for each token.
            return_length = False,
            verbose = True,
            return_coord_ids: bool = True,
            **kwargs
        ):

        result = super().__call__(
            text,
            text_pair,
            text_target,
            text_pair_target,
            add_special_tokens,
            padding, 
            truncation,
            max_length,
            stride,
            is_split_into_words,
            pad_to_multiple_of,
            padding_side,
            return_tensors,
            return_token_type_ids,
            return_attention_mask,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_offsets_mapping,
            return_length,
            verbose,
            **kwargs
        )

        if return_coord_ids:
            result["coord_indices"] = self.coord_indices(result)

        return result
    

    def coord_indices(self, input_ids):
        # Coord token mask
        mask = super().is_coord_token(input_ids)

        coord_indices = self._get_output_tensor(input_ids)

        for i, l in enumerate(coord_indices):
            is_even = 0
            for j in range(len(l)):
                if mask[i][j]:
                    coord_indices[i][j] = is_even + 1
                    is_even = (is_even + 1) % 2
                else:
                    is_even = 0
        
        return coord_indices
    

    def pad(
            self,
            encoded_inputs,
            padding = True,
            max_length = None,
            pad_to_multiple_of = None,
            padding_side = None,
            return_attention_mask = None,
            return_tensors = None,
            verbose = True
        ):

        # TODO: fix padding
        return super().pad(
            encoded_inputs,
            padding, max_length,
            pad_to_multiple_of,
            padding_side,
            return_attention_mask,
            return_tensors,
            verbose
        )

    
    @singledispatchmethod
    def _get_output_tensor(self, mask):
        raise TypeError(f"Invalid type: {type(mask)}")
    
    @_get_output_tensor.register
    def _(self, mask: torch.Tensor):
        return torch.zeros_like(mask)
    
    @_get_output_tensor.register
    def _(self, mask: list):
        return [[0] * len(i) for i in mask]
