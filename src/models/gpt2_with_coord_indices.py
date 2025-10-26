import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


class GPT2ModelWithCoordIndices(GPT2LMHeadModel):
    """
    This class extends GPT2LMHeadModel to include additional coordinates positional embeddings
    """

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.coord_index_embd = nn.Embedding(3, config.n_embd)

        self.post_init()


    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            coord_indices=None,
            labels=None,
            inputs_embeds=None,
            **kwargs
        ):

        assert inputs_embeds is None

        index_embeds = self.coord_index_embd(coord_indices)
        tokens_embeds = self.transformer.wte(input_ids)

        input_embeds = tokens_embeds + index_embeds

        # TODO: fix kwargs handing
        return super().forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            # **kwargs
        )
