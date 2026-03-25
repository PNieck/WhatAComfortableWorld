"""GPT2 model with XY indices for coordinate position embeddings.

This module extends CustomGPT2 to incorporate XY coordinate indices as
additional embeddings for improved floor plan coordinate generation.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config
from .custom_gpt2 import CustomGPT2

import src.tokens as tokens


class GPT2ModelWithXYIndices(CustomGPT2):
    """
    GPT2 model with XY indices embeddings for coordinate generation.
    
    Extends CustomGPT2 by adding XY coordinate positional embeddings to distinguish
    between X and Y coordinates in floor plan generation.
    """

    def __init__(self, config: GPT2Config):
        """
        Initializes the GPT2ModelWithXYIndices.
        
        Creates a GPT2 model with coordinate index embeddings for tracking X/Y coordinate
        positions during floor plan generation.
        
        :param config: GPT2 model configuration
        """
        super().__init__(config)

        self.coord_index_embd = nn.Embedding(3, config.n_embd)

        self.use_xy_indices = True
        self._last_xy_indices = None

        self.post_init()


    def forward(
            self,
            input_ids,
            attention_mask=None,
            xy_indices=None,
            labels=None,
            inputs_embeds=None,
            use_cache=None,
            **kwargs
        ):
        """
        Forward pass through the model with XY indices embeddings.
        
        Processes input IDs with coordinate index embeddings. If XY indices are not
        provided, they are computed from input IDs or retrieved from cache during generation.
        
        :param input_ids: Token IDs for the input sequence
        :param attention_mask: Attention mask (optional)
        :param xy_indices: Pre-computed XY indices (optional). If None, they will be computed
        :param labels: Target labels for training (optional)
        :param inputs_embeds: Pre-computed input embeddings (optional)
        :param use_cache: Whether to use caching during generation
        :param kwargs: Additional arguments to pass to parent forward method
        :return: Model output from parent class with XY embeddings applied
        """

        if self.use_xy_indices:
            if xy_indices is None:
                if use_cache is True:
                    xy_indices = self.xy_indices_from_cache(input_ids)

                else:
                    coord_indices, mask = self.coord_indices(input_ids)
                    xy_indices = self.xy_indices(coord_indices, mask)

            index_embeds = self.coord_index_embd(xy_indices)

            if inputs_embeds is None:
                tokens_embeds = self.transformer.wte(input_ids)
                inputs_embeds = tokens_embeds + index_embeds
            else:
                inputs_embeds += index_embeds

            if use_cache is True:
                self.update_xy_indices_cache(xy_indices)

        result = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            **kwargs
        )

        return result
    

    def xy_indices(self, coord_indices: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts XY indices from coordinate indices.
        
        Determines whether each coordinate is an X or Y component by computing modulo 2
        of the coordinate index. Maps 0,2,4... to X index 1, and 1,3,5... to Y index 2.
        
        :param coord_indices: Coordinate indices
        :param coord_mask: Boolean mask indicating which positions are coordinates
        :return: XY indices (1 for X coordinates, 2 for Y coordinates)
        """
        with torch.no_grad():
            coord_indices[coord_mask] = ((coord_indices[coord_mask]-1) % 2) + 1

            return coord_indices
    

    def coord_indices(self, input_ids: torch.Tensor):
        """
        Computes coordinate indices from input token IDs.
        
        Identifies which coordinate each token represents based on the cumulative count
        of coordinate tokens in the sequence. Handles both 1D and 2D (batched) input.
        
        :param input_ids: Input token IDs
        :return: Tuple of (coord_indices, coord_mask) where coord_indices tracks coordinate position
                 and coord_mask indicates which tokens are coordinates
        """
        with torch.no_grad():
            coord_mask = tokens.is_coord(input_ids)
            neg_mask = ~coord_mask

            if coord_mask.dim() == 1:
                c = torch.cumsum(neg_mask)
                d = torch.diff(c[neg_mask], 1, -1, torch.tensor([0]))
                e = coord_mask.type(torch.int)
                e[neg_mask] = -d
                result = torch.cumsum(e)

                return result

            assert coord_mask.dim() == 2

            c = torch.cumsum(coord_mask, 1)

            batch =  input_ids.shape[0]
            neg_sum = torch.sum(neg_mask)
            c_n_len = neg_sum + batch + 1
            c_n = torch.zeros(c_n_len, dtype=torch.long, device=input_ids.device)

            rows = torch.nonzero(neg_mask)[:, 0]
            indices = rows + torch.arange(neg_sum, device=input_ids.device) + 1
            
            c_n[indices] = c[neg_mask]
            
            d = torch.diff(c_n)

            e = coord_mask.type(torch.long)
            e[neg_mask] = -d[indices-1]

            result = torch.cumsum(e, 1)

            return (result, coord_mask)
        
            
    def xy_indices_from_cache(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes XY indices using cached values during generation.
        
        Uses the previous XY index to determine the current one during autoregressive
        generation. Alternates between X (1) and Y (2) indices for consecutive coordinates.
        
        :param input_ids: Input token IDs (single token during generation)
        :return: XY indices for the current generation step
        """
        if input_ids.shape[1] > 1:
            coord_indices, mask = self.coord_indices(input_ids)
            return self.xy_indices(coord_indices, mask)
        
        coord_mask = tokens.is_coord(input_ids).squeeze()

        xy_indices = torch.zeros(input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)

        x_coord_mask = ((self._last_xy_indices == 0) | (self._last_xy_indices == 2)) & coord_mask
        xy_indices[x_coord_mask] = 1

        y_coord_mask = (self._last_xy_indices == 1) & coord_mask
        xy_indices[y_coord_mask] = 2

        return xy_indices.unsqueeze(1)
    

    def update_xy_indices_cache(self, xy_indices: torch.Tensor):
        """
        Updates the cached XY index after each generation step.
        
        Stores the most recent XY index for use in determining the next coordinate type
        during autoregressive generation.
        
        :param xy_indices: XY indices from the current generation step
        """
        self._last_xy_indices = xy_indices[:, -1]

