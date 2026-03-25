import re

from transformers import PreTrainedTokenizer

import src.tokens as tokens


class FloorPlanTokenizer(PreTrainedTokenizer):
    """
    Tokenizer for floor plan sequences.

    Handles mapping between textual tokens used to describe floor plans and
    integer IDs expected by models. The tokenizer includes special tokens and
    generates coordinate and room tokens.
    """

    def __init__(self, res=256, *args, **kwargs):
        """
        Initialize the tokenizer.

        :param res: Coordinate resolution (number of discrete coordinate tokens)
        """
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
        """Return number of tokens in the tokenizer vocabulary.

        Includes special tokens and coordinate/room tokens.
        :return: Vocabulary size (int)
        """
        return len(self.token2id)


    def _tokenize(self, text):
        """Split a sequence string into individual tokens.

        Tokens are expected to have the form `<...>`. This method extracts
        those tokens preserving their delimiters.

        :param text: Input sequence string
        :return: List of token strings
        """
        return re.findall(r'[^>]+>', text)


    def _convert_token_to_id(self, token):
        """Convert a single token string to its integer ID.

        :param token: Single token string
        :return: Integer token ID
        """
        return self.token2id[token]


    def _convert_id_to_token(self, index):
        """Convert an integer token ID back to its token string.

        :param index: Integer token ID
        :return: Token string
        """
        return self.id2token[index]


    def convert_tokens_to_string(self, tokens):
        """Join characters back into text."""
        return "".join(tokens)


    def build_inputs_with_special_tokens(self, token_ids, _):
        """Wrap input in start and end sequence tokens."""
        return [tokens.START_SEQ_TOKEN_ID] + token_ids + [tokens.END_SEQ_TOKEN_ID]


    def get_vocab(self):
        """Return the tokenizer vocabulary mapping token->id.

        :return: Dictionary mapping token strings to integer IDs
        """
        return self.token2id

    @property
    def vocab_size(self) -> int:
        """Return the number of non-special tokens in the vocabulary.

        :return: Integer count of non-special tokens
        """
        return len(self.token2id) - 4


    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Placeholder: saving vocabulary is not implemented.

        :param save_directory: Directory to save vocabulary files
        :param filename_prefix: Optional filename prefix
        """
        pass


    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        # For GPT-style models, we don’t use sentence pairs.
        # So we just return a list of 0s with the same length as input_ids.
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)  # +2 for <s> and </s>
        else:
            # If you ever add sentence-pair mode
            return [0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 1)
