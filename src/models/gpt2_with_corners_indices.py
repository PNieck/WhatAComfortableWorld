from .gpt2_with_xy_indices import GPT2ModelWithXYIndices
from transformers import GPT2Config
import torch.nn as nn
import torch


class GPT2ModelWithCornerIndices(GPT2ModelWithXYIndices):
    MAX_CORNERS = 58

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.corner_embd = nn.Embedding(self.MAX_CORNERS+1, config.n_embd)
        self._last_corner_indices = None
        self._second_to_last_corner_indices = None

        self.post_init()


    def forward(
            self,
            input_ids,
            attention_mask=None,
            corner_indices=None,
            labels=None,
            inputs_embeds=None,
            use_cache=None,
            **kwargs
        ):

        assert inputs_embeds is None

        if corner_indices is None:
            if use_cache is True:
                corner_indices, xy_indices = self.corner_and_xy_indices_from_cache(input_ids)

            else:
                corner_indices, xy_indices = self.corner_and_xy_indices(input_ids)

        else:
            xy_indices = None

        too_big_corners_indices = corner_indices < self.MAX_CORNERS
        if torch.any(too_big_corners_indices):
            corner_indices = torch.where(too_big_corners_indices, corner_indices, self.MAX_CORNERS-1)
            print("Warning: too big corner indices")

        inputs_embeds = self.transformer.wte(input_ids)
        inputs_embeds += self.corner_embd(corner_indices)

        if use_cache is True:
            self.update_corner_indices_cache(corner_indices)

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            xy_indices=xy_indices,
            use_cache=use_cache,
            **kwargs
        )


    def corner_and_xy_indices(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coord_indices, coord_mask = self.coord_indices(input_ids)

        corner_indices = coord_indices.clone()
        corner_indices = self.corner_indices(corner_indices, coord_mask)

        xy_indices = self.xy_indices(coord_indices, coord_mask)

        return corner_indices, xy_indices
    

    def corner_indices(self, coord_indices: torch.Tensor, coord_mask: torch.Tensor):
        with torch.no_grad():
            dtype = coord_indices.dtype
            coord_indices[coord_mask] = torch.ceil(coord_indices[coord_mask] / 2).type(dtype)

            return coord_indices
        

    def corner_and_xy_indices_from_cache(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.shape[1] > 1:
            return self.corner_and_xy_indices(input_ids)

        coord_mask = self.coord_mask(input_ids).squeeze()
        corner_indices = torch.zeros(input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)

        # First corner
        first_corner_mask = coord_mask & (self._last_corner_indices == 0)
        corner_indices[first_corner_mask] = 1

        last_two_same_corner_mask = self._last_corner_indices == self._second_to_last_corner_indices

        # Next corner
        next_corner_mask = last_two_same_corner_mask & (~first_corner_mask) & coord_mask
        corner_indices[next_corner_mask] = self._last_corner_indices[next_corner_mask] + 1

        # Same corner as last time
        same_corner_mask = (~last_two_same_corner_mask) & (~first_corner_mask) & coord_mask
        corner_indices[same_corner_mask] = self._last_corner_indices[same_corner_mask]

        return corner_indices.unsqueeze(1), None


    def update_corner_indices_cache(self, corner_indices: torch.Tensor):
        if corner_indices.shape[1] >= 2:
            self._last_corner_indices = corner_indices[:, -1]
            self._second_to_last_corner_indices = corner_indices[:, -2]
        else:
            self._second_to_last_corner_indices = self._last_corner_indices
            self._last_corner_indices = corner_indices[:, -1]

