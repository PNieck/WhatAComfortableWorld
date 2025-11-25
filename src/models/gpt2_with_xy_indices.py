import torch
import torch.nn as nn
from transformers import GPT2Config
from .gpt2 import CustomGPT2

import tokens


class GPT2ModelWithXYIndices(CustomGPT2):
    """
    This class extends GPT2LMHeadModel to include additional coordinates positional embeddings
    """

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.coord_index_embd = nn.Embedding(3, config.n_embd)

        self.min_coord_token_id = tokens.coord_token_id(0)
        self.max_coord_token_id = tokens.coord_token_id(256)

        self.post_init()


    # TODO: implement caching for inference
    def forward(
            self,
            input_ids,
            attention_mask=None,
            xy_indices=None,
            labels=None,
            inputs_embeds=None,
            **kwargs
        ):

        if xy_indices is None:
            coord_indices, mask = self.coord_indices(input_ids)
            xy_indices = self.xy_indices(coord_indices, mask)

        index_embeds = self.coord_index_embd(xy_indices)

        if inputs_embeds is None:
            tokens_embeds = self.transformer.wte(input_ids)
            inputs_embeds = tokens_embeds + index_embeds

        else:
            inputs_embeds += index_embeds

        # TODO: fix kwargs handing
        result = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            # **kwargs
        )

        return result
    

    def xy_indices(self, coord_indices: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            coord_indices[coord_mask] = ((coord_indices[coord_mask]-1) % 2) + 1

            return coord_indices
    

    def coord_indices(self, input_ids: torch.Tensor):
        with torch.no_grad():
            coord_mask = (input_ids >= self.min_coord_token_id) & (input_ids <= self.max_coord_token_id)
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

