from .gpt2_with_xy_indices import GPT2ModelWithXYIndices
from transformers import GPT2Config, GPT2LMHeadModel
import torch.nn as nn
import torch


class GPT2ModelWithCornerIndices(GPT2ModelWithXYIndices):
    MAX_CORNERS = 58

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.corner_embd = nn.Embedding(self.MAX_CORNERS+1, config.n_embd)

        self.post_init()


    def forward(
            self,
            input_ids,
            attention_mask=None,
            corner_indices=None,
            labels=None,
            inputs_embeds=None,
            **kwargs
        ):

        assert inputs_embeds is None

        if corner_indices is None:
            coord_indices, mask = self.coord_indices(input_ids)

            corner_indices = coord_indices.clone()
            corner_indices = self.corner_indices(corner_indices, mask)

            xy_indices = self.xy_indices(coord_indices, mask)
        else:
            xy_indices = None

        inputs_embeds = self.transformer.wte(input_ids)
        inputs_embeds += self.corner_embd(corner_indices)

        # TODO: fix kwargs handing
        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            xy_indices=xy_indices
        )


    def corner_indices(self, coord_indices: torch.Tensor, coord_mask: torch.Tensor):
        with torch.no_grad():
            dtype = coord_indices.dtype
            coord_indices[coord_mask] = torch.ceil(coord_indices[coord_mask] / 2).type(dtype)

            return coord_indices

