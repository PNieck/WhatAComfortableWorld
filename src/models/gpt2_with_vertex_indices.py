"""GPT2 model with corner indices for floor plan generation.

This module extends the GPT2ModelWithXYIndices to incorporate corner indices
as additional embeddings for improved floor plan generation.
"""

from .gpt2_with_xy_indices import GPT2ModelWithXYIndices
from transformers import GPT2Config
import torch.nn as nn
import torch

import src.tokens as tokens


class GPT2ModelWithVertexIndices(GPT2ModelWithXYIndices):
    """
    GPT2 model with corner indices embeddings for floor plan generation.
    
    Extends GPT2ModelWithXYIndices by adding corner index embeddings to input embeddings.
    Tracks corner indices during generation to provide contextual information about
    which corner of the floor plan is being generated.
    """
    MAX_CORNERS = 58

    def __init__(self, config: GPT2Config):
        """
        Initializes the GPT2ModelWithCornerIndices.
        
        Creates a GPT2 model with corner index embeddings for tracking corner positions
        during floor plan generation.
        
        :param config: GPT2 model configuration
        """
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
        """
        Forward pass through the model with corner indices embeddings.
        
        Processes input IDs with corner indices embeddings. If corner indices are not
        provided, they are computed from input IDs or retrieved from cache during generation.
        
        :param input_ids: Token IDs for the input sequence
        :param attention_mask: Attention mask (optional)
        :param corner_indices: Pre-computed corner indices (optional). If None, they will be computed.
        :param labels: Target labels for training (optional)
        :param inputs_embeds: Pre-computed input embeddings (optional). Must be None
        :param use_cache: Whether to use caching during generation
        :param kwargs: Additional arguments to pass to parent forward method
        :return: Model output from parent class with corner embeddings applied
        """

        assert inputs_embeds is None

        if corner_indices is None:
            if use_cache is True:
                corner_indices, xy_indices = self.corner_and_xy_indices_from_cache(input_ids)

            else:
                corner_indices, xy_indices = self.corner_and_xy_indices(input_ids)

        else:
            xy_indices = None

        too_big_corners_indices = corner_indices >= self.MAX_CORNERS
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
        """
        Computes corner and XY indices from input token IDs.
        
        Extracts coordinate indices from input IDs, then computes corner indices
        (which corner of the polygon each coordinate belongs to) and XY indices
        (X or Y component of each coordinate).
        
        :param input_ids: Input token IDs
        :return: Tuple of (corner_indices, xy_indices)
        """
        coord_indices, coord_mask = self.coord_indices(input_ids)

        corner_indices = coord_indices.clone()
        corner_indices = self.corner_indices(corner_indices, coord_mask)

        xy_indices = self.xy_indices(coord_indices, coord_mask)

        return corner_indices, xy_indices
    

    def corner_indices(self, coord_indices: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts corner indices from coordinate indices.
        
        Computes which corner each coordinate belongs to by dividing the coordinate
        index by 2 (since each corner has 2 coordinates: X and Y).
        
        :param coord_indices: Coordinate indices
        :param coord_mask: Boolean mask indicating which positions are coordinates
        :return: Corner indices derived from coordinate indices
        """
        with torch.no_grad():
            dtype = coord_indices.dtype
            coord_indices[coord_mask] = torch.ceil(coord_indices[coord_mask] / 2).type(dtype)

            return coord_indices
        

    def corner_and_xy_indices_from_cache(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes corner and XY indices using cached values during generation.
        
        Uses previously cached corner indices to efficiently determine the next corner
        index during autoregressive generation.
        
        :param input_ids: Input token IDs (single token during generation)
        :return: Tuple of (corner_indices, None). XY indices are None as they're not needed during caching
        """
        if input_ids.shape[1] > 1:
            return self.corner_and_xy_indices(input_ids)

        coord_mask = tokens.is_coord(input_ids).squeeze()
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


    def update_corner_indices_cache(self, corner_indices: torch.Tensor) -> None:
        """
        Updates the cached corner indices after each generation step.
        
        Maintains a rolling window of the last two corner indices for use in the next
        generation step. This enables logic to determine when we've moved to a new corner.
        
        :param corner_indices: Corner indices from the current generation step
        """
        if corner_indices.shape[1] >= 2:
            self._last_corner_indices = corner_indices[:, -1]
            self._second_to_last_corner_indices = corner_indices[:, -2]
        else:
            self._second_to_last_corner_indices = self._last_corner_indices
            self._last_corner_indices = corner_indices[:, -1]

