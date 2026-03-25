"""Custom GPT-2 model for floor plan sequence generation.

Provides customized GPT-2 architecture with floor plan-specific tokenization,
coordinate generation, and validation logic.
"""

from enum import Enum
from typing import List

import shapely.validation
import torch
import shapely
from shapely.ops import linemerge
from transformers import GPT2Config, GPT2LMHeadModel

import src.tokens as tokens
from src.floor_plan_tokenizer import FloorPlanTokenizer
from src.floor_plan import RoomType
from src.sequence.from_sequence import boundary_from_sequence
from src.geom_utils import LineType, line_type, create_line
from src.training_config import TrainingConfig


def get_gpt2_config(config: TrainingConfig) -> GPT2Config:
    """Create GPT2 configuration from training configuration.
    
    :param config: Training configuration containing model parameters
    :return: Configured GPT2Config object
    """
    return GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.max_sequence_len,
        n_ctx=config.max_sequence_len,
        n_layer=config.model_layers_cnt,
        n_head=config.model_heads_cnt,
        n_embd=config.model_embedding_dim,
        bos_token_id=tokens.START_SEQ_TOKEN_ID,
        eos_token_id=tokens.END_SEQ_TOKEN_ID,
    )


class TokenToGenerateType(Enum):
    """Token type enumeration for floor plan sequence generation.
    
    Represents each step in the room coordinate generation sequence, including
    room type selection, corner coordinate generation, and end tokens.
    """
    FirstRoom = 0

    FirstXInRoom = 1
    FirstYInRoom = 2

    SecondXInRoom = 3
    SecondYInRoom = 4

    ThirdXInRoom = 5
    ThirdYInRoom = 6

    FourthXInRoom = 7
    FourthYInRoom = 8

    NthXInRoom = 9
    NthYInRoom = 10

    EndToken = 11


class CustomGPT2(GPT2LMHeadModel):
    """
    Custom GPT2 model for floor plan generation with optional constrained token masking.
    """
    
    def __init__(self, config: GPT2Config):
        """
        Initializes the CustomGPT2 model with configuration and state tracking.
        
        :param config: GPT2 configuration
        :type config: GPT2Config
        """
        self.max_seq_len = config.n_embd
        self.use_masked_inference = False

        super().__init__(config)

        self.cached_remaining_empty_spaces: List[shapely.Polygon] = None
        self.cached_room_in_generation: List[List[int]] = None

        self.cached_prev_token_to_gen: torch.Tensor = None

        self.tokenizer = FloorPlanTokenizer()

        self.validity_problems = 0
    
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs
    ):
        """
        Forward pass with optional masked inference for constrained generation.
        
        If masked inference is enabled and not in training mode, applies geometric constraints
        to the logits to ensure only valid tokens can be selected.
        
        :param input_ids: Input token IDs
        :param inputs_embeds: Input embeddings
        :param use_cache: Whether to use key-value caching
        :return: Model output with optionally masked logits
        """
        local_input_ids = input_ids

        if inputs_embeds is not None:
            input_ids = None

        output = super().forward(
            input_ids=input_ids,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        if self.use_masked_inference and not self.training:
            output.logits = self.mask_logits_for_inference(local_input_ids, output.logits, use_cache)

        return output
    

    def mask_logits_for_inference(self, input_ids: torch.Tensor, logits: torch.Tensor, use_cache: bool):
        """
        Applies geometric constraints to logits based on current generation state.
        
        Determines the next token type to generate and applies appropriate masking to ensure
        only valid spatial positions are allowed.
        
        :param input_ids: Input token IDs
        :type input_ids: torch.Tensor
        :param logits: Model output logits to be masked
        :type logits: torch.Tensor
        :param use_cache: Whether key-value caching is enabled
        :type use_cache: bool
        :return: Masked logits with invalid tokens set to -inf
        :rtype: torch.Tensor
        """

        assert use_cache == True, "Masked inference requires cache to be enabled"
        assert input_ids is not None

        for batch_no in range(logits.shape[0]):
            token_to_generate = self.get_token_to_generate_type(input_ids, batch_no)

            match token_to_generate:
                case TokenToGenerateType.FirstRoom:
                    self.mask_for_first_room(logits)
                    self.init_cache(input_ids)
                    break

                case TokenToGenerateType.FirstXInRoom:
                    self.mask_for_first_x_in_room(logits, batch_no)

                case TokenToGenerateType.FirstYInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_first_y_in_room(logits, batch_no)

                case TokenToGenerateType.SecondXInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_second_x_in_room(logits, batch_no)

                case TokenToGenerateType.SecondYInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_second_y_in_room(logits, batch_no)

                case TokenToGenerateType.ThirdXInRoom | TokenToGenerateType.FourthXInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_nth_x_in_room(logits, batch_no)
                    self.invalidate_room_end(logits, batch_no)

                case TokenToGenerateType.ThirdYInRoom | TokenToGenerateType.FourthYInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_nth_y_in_room(logits, batch_no)
                    self.invalidate_room_end(logits, batch_no)

                case TokenToGenerateType.NthXInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_nth_x_in_room(logits, batch_no)
                    self.check_possible_room_end(logits, batch_no)

                case TokenToGenerateType.NthYInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_nth_y_in_room(logits, batch_no)
                    self.invalidate_room_end(logits, batch_no)

                case TokenToGenerateType.EndToken:
                    self.mask_for_end_token(logits, batch_no)


            self.cached_prev_token_to_gen[batch_no] = token_to_generate.value
            
        return logits


    def get_token_to_generate_type(self, input_ids: torch.Tensor, batch_no) -> TokenToGenerateType:
        """
        Determines the type of token that should be generated next.
        
        Based on the current input and generation state, determines whether the next token
        should be a room type, coordinate, or end token.
        
        :param input_ids: Current input token IDs
        :type input_ids: torch.Tensor
        :param batch_no: Batch index
        :return: The type of token to generate next
        :rtype: TokenToGenerateType
        """
        if input_ids.shape[1] > 1:                  # We are assuming using kv caching
            return TokenToGenerateType.FirstRoom
        
        id = input_ids[batch_no, 0]
        if self.room_generation_is_finished(id, batch_no):
            self.finish_room_generation(batch_no)
            if self.cached_remaining_empty_spaces[batch_no].is_empty:
                return TokenToGenerateType.EndToken

            return TokenToGenerateType.FirstXInRoom
        
        cached_token = self.cached_prev_token_to_gen[batch_no].item()
        if cached_token == TokenToGenerateType.EndToken.value:
            return TokenToGenerateType.EndToken
        
        result = TokenToGenerateType(cached_token + 1)
        if result.value == TokenToGenerateType.EndToken.value:
            result = TokenToGenerateType.NthXInRoom
            
        return result


    def room_generation_is_finished(self, id, batch_no: int) -> bool:
        """
        Checks if room generation is complete.
        
        A room is finished when next room start token was generated or room has been closed
        (first corner equals last corner) and at least 10 coordinates have been collected.
        
        :param id: Current token ID
        :param batch_no: Batch index
        :type batch_no: int
        :return: True if room generation is complete
        :rtype: bool
        """
        if tokens.is_room(id):
            return True

        coords = self.cached_room_in_generation[batch_no]
        if len(coords) < 10:
            return False
        
        first_corner = coords[:2]
        last_corner = coords[-2:]

        return first_corner[0] == last_corner[0] and first_corner[1] == last_corner[1]


    def mask_for_first_room(self, logits: torch.Tensor):
        """
        Masks logits for the first room token selection.
        
        Only allows room type tokens and disallows special tokens, coordinates, and boundary tokens.
        
        :param logits: Model output logits to mask
        :type logits: torch.Tensor
        """
        logits[:, :, tokens.START_SEQ_TOKEN_ID] = -torch.inf
        logits[:, :, tokens.UNK_TOKEN_ID] = -torch.inf
        logits[:, :, tokens.PAD_TOKEN_ID] = -torch.inf
        logits[:, :, tokens.BOUNDARY_TOKEN_ID] = -torch.inf
        logits[:, :, tokens.DOOR_TOKEN_ID] = -torch.inf

        logits[:, :, tokens.MIN_COORD_ID:tokens.MAX_COORD_ID] = -torch.inf
        logits[:, :, tokens.END_SEQ_TOKEN_ID] = -torch.inf


    def mask_for_first_x_in_room(self, logits, batch_no):
        """
        Masks logits for the first X coordinate of a room.
        
        Restricts X coordinate choices to the bounds of the available empty space.
        
        :param logits: Model output logits to mask
        :param batch_no: Batch index
        """
        empty_space = self.cached_remaining_empty_spaces[batch_no]

        is_valid = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        match empty_space.geom_type:
            case "Polygon":               
                (min_x, max_x) = self.geometry_x_bounds(empty_space)

                min_token = tokens.coord_token_id(min_x)
                max_token = tokens.coord_token_id(max_x)

                is_valid[min_token:(max_token+1)] = True

            case "MultiPolygon":
                for polygon in empty_space.geoms:
                    (min_x, max_x) = self.geometry_x_bounds(polygon)

                    min_token = tokens.coord_token_id(min_x)
                    max_token = tokens.coord_token_id(max_x)

                    is_valid[min_token:(max_token+1)] = True

            case _:
                raise Exception(f"Unexpected geometry type {empty_space.geom_type}")

        logits[batch_no, :, ~is_valid] = -torch.inf

    
    def mask_for_first_y_in_room(self, logits, batch_no):
        """
        Masks logits for the first Y coordinate of a room.
        
        Restricts Y coordinates to those that form valid line intersections with the empty space
        at the chosen X coordinate.
        
        :param logits: Model output logits to mask
        :param batch_no: Batch index
        """
        empty_space = self.cached_remaining_empty_spaces[batch_no]
        (min_y, max_y) = self.geometry_y_bounds(empty_space)

        x = self.cached_room_in_generation[batch_no][0]

        line = create_line((x, min_y), (x, max_y))
        intersection = empty_space.intersection(line)
        if intersection.geom_type == "MultiLineString":
            intersection = linemerge(intersection)

        is_valid = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        assert not intersection.is_empty, "Invalid input boundary" 

        match intersection.geom_type:
            case "LineString":
                (min_y, max_y) = self.geometry_y_bounds(intersection)

                min_token = tokens.coord_token_id(min_y)
                max_token = tokens.coord_token_id(max_y)

                is_valid[min_token:(max_token+1)] = True

            case "MultiLineString":
                for line in intersection.geoms:
                    (min_y, max_y) = self.geometry_y_bounds(line)

                    min_token = tokens.coord_token_id(min_y)
                    max_token = tokens.coord_token_id(max_y)

                    is_valid[min_token:(max_token+1)] = True

            case _:
                raise Exception("Unexpected geometry type")

        logits[batch_no, :, ~is_valid] = -torch.inf


    def mask_for_second_x_in_room(self, logits, batch_no):
        """
        Masks logits for the second X coordinate of a room.
        
        Constrains the second X coordinate based on horizontal line intersections with empty space.
        
        :param logits: Model output logits to mask
        :param batch_no: Batch index
        """
        first_x = self.cached_room_in_generation[batch_no][0]
        first_y = self.cached_room_in_generation[batch_no][1]

        first_point = shapely.Point(first_x, first_y)

        empty_space = self._get_empty_space(batch_no)
        (min_x, max_x) = self.geometry_x_bounds(empty_space)

        is_valid = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        line = create_line((min_x, first_y), (max_x, first_y))
        inter = empty_space.intersection(line)
        self.mask_x_direction(inter, first_point, is_valid)
                
        logits[batch_no, :, ~is_valid] = -torch.inf


    def mask_for_second_y_in_room(self, logits, batch_no):
        """
        Masks logits for the second Y coordinate of a room.
        
        Constrains Y coordinate based on vertical line intersections, preventing staying at same position.
        
        :param logits: Model output logits to mask
        :param batch_no: Batch index
        """
        first_x = self.cached_room_in_generation[batch_no][0]
        first_y = self.cached_room_in_generation[batch_no][1]

        last_x = self.cached_room_in_generation[batch_no][2]
        last_y = first_y

        last_y_token_id = tokens.coord_token_id(last_y)

        is_valid = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        if last_x != first_x:
            is_valid[last_y_token_id] = True

        else:
            first_point = shapely.Point(first_x, first_y)

            empty_space = self._get_empty_space(batch_no)
            (min_y, max_y) = self.geometry_y_bounds(empty_space)

            line = create_line((last_x, min_y), (last_x, max_y))
            inter = empty_space.intersection(line)
            self.mask_y_direction(inter, first_point, is_valid)

            # We cannot stay in one place
            is_valid[last_y_token_id] = False     

        logits[batch_no, :, ~is_valid] = -torch.inf


    def mask_for_nth_x_in_room(self, logits, batch_no):
        """
        Masks logits for subsequent X coordinates in a room (3rd+ coordinates).
        
        Applies geometric constraints based on the direction of previous line segments and
        available space.
        
        :param logits: Model output logits to mask
        :param batch_no: Batch index
        """
        second_to_last_x = self.cached_room_in_generation[batch_no][-4]
        second_to_last_y = self.cached_room_in_generation[batch_no][-3]

        last_x = self.cached_room_in_generation[batch_no][-2]
        last_y = self.cached_room_in_generation[batch_no][-1]

        last_line_type = line_type(second_to_last_x, second_to_last_y, last_x, last_y)

        is_valid = self.get_is_valid_tensor_with_rooms_as_valid()

        last_point = shapely.Point(last_x, last_y)

        empty_space_polygon = self._get_empty_space(batch_no)
        (min_x, max_x) = self.geometry_x_bounds(empty_space_polygon)

        current_room_linestring = self.get_linestring_of_current_room(batch_no)

        match last_line_type:
            case LineType.Vertical:
                line = create_line((min_x, last_y), (max_x, last_y))
                inter = empty_space_polygon.intersection(line)
                if inter.geom_type != "Point":
                    inter = inter.difference(current_room_linestring)

                self.mask_x_direction(inter, last_point, is_valid)

                # We cannot stay in the same place
                # token_id = tokens.coord_token_id(last_x)
                # is_valid[token_id] = False

                # Checking if we can stay in the same x
                if second_to_last_y < last_y:
                    next_y = last_y + 1
                else:
                    next_y = last_y - 1

                next_point = shapely.Point(last_x, next_y)
                if not empty_space_polygon.intersects(next_point):
                    token_id = tokens.coord_token_id(last_x)
                    is_valid[token_id] = False

            case LineType.Horizontal:
                if second_to_last_x < last_x:
                    line = create_line((last_x, last_y), (max_x, last_y))
                    inter = empty_space_polygon.intersection(line)
                    if inter.geom_type != "Point":
                        inter = inter.difference(current_room_linestring)
                    self.mask_x_direction(inter, last_point, is_valid)

                else:
                    line = create_line((min_x, last_y), (last_x, last_y))
                    inter = empty_space_polygon.intersection(line)
                    if inter.geom_type != "Point":
                        inter = inter.difference(current_room_linestring)
                    self.mask_x_direction(inter, last_point, is_valid)

            case _:
                raise Exception("Unexpected line type")            

        logits[batch_no, :, ~is_valid] = -torch.inf


    def mask_for_nth_y_in_room(self, logits, batch_no):
        """
        Masks logits for subsequent Y coordinates in a room (3rd+ coordinates).
        
        Applies geometric constraints based on vertical line intersections and room topology.
        
        :param logits: Model output logits to mask
        :param batch_no: Batch index
        """
        second_to_last_x = self.cached_room_in_generation[batch_no][-5]
        second_to_last_y = self.cached_room_in_generation[batch_no][-4]

        last_x = self.cached_room_in_generation[batch_no][-3]
        last_y = self.cached_room_in_generation[batch_no][-2]

        prev_x = self.cached_room_in_generation[batch_no][-1]

        is_valid = self.get_is_valid_tensor_with_rooms_as_valid()

        second_y_token_id = tokens.coord_token_id(last_y)

        if prev_x != last_x:
            is_valid[second_y_token_id] = True

        else:
            last_point = shapely.Point(last_x, last_y)

            empty_space_polygon = self._get_empty_space(batch_no)
            (min_y, max_y) = self.geometry_y_bounds(empty_space_polygon)

            current_room_linestring = self.get_linestring_of_current_room(batch_no)

            last_line_type = line_type(second_to_last_x, second_to_last_y, last_x, last_y)
            assert last_line_type != LineType.Diagonal

            match last_line_type:
                case LineType.Horizontal:
                    line = create_line((prev_x, min_y), (prev_x, max_y))
                    inter = empty_space_polygon.intersection(line)
                    if inter.geom_type != "Point":
                        inter = inter.difference(current_room_linestring)
                    self.mask_y_direction(inter, last_point, is_valid)

                case LineType.Vertical:
                    if second_to_last_y < last_y:
                        line = create_line((prev_x, last_y), (prev_x, max_y))
                        inter = empty_space_polygon.intersection(line)
                        if inter.geom_type != "Point":
                            inter = inter.difference(current_room_linestring)
                        self.mask_y_direction(inter, last_point, is_valid)

                    else:
                        line = create_line((prev_x, last_y), (prev_x, min_y))
                        inter = empty_space_polygon.intersection(line)
                        if inter.geom_type != "Point":
                            inter = inter.difference(current_room_linestring)
                        self.mask_y_direction(inter, last_point, is_valid)

            # We cannot stay in the same place
            is_valid[second_y_token_id] = False

        logits[batch_no, :, ~is_valid] = -torch.inf


    def mask_for_end_token(self, logits, batch_no):
        """
        Masks logits to allow only the end token.
        
        Used when no more space is available for additional rooms.
        
        :param logits: Model output logits to mask
        :param batch_no: Batch index
        """
        mask = torch.ones(len(self.tokenizer), dtype=torch.bool)
        mask[tokens.END_SEQ_TOKEN_ID] = False

        logits[batch_no, :, mask] = -torch.inf


    def invalidate_room_end(self, logits, batch_no):
        """
        Prevents ending the current room by masking room type tokens.
        
        Used to enforce minimum room complexity before allowing room completion.
        
        :param logits: Model output logits to mask
        :param batch_no: Batch index
        """
        logits[batch_no, :, tokens.MIN_ROOM_ID:(tokens.MAX_ROOM_ID+1)] = -torch.inf


    def check_possible_room_end(self, logits, batch_no):
        """
        Conditionally masks room end tokens based on geometric validity.
        
        Prevents ending rooms with diagonal lines, which are geometrically invalid.
        
        :param logits: Model output logits to mask
        :param batch_no: Batch index
        """
        first_x = self.cached_room_in_generation[batch_no][0]
        first_y = self.cached_room_in_generation[batch_no][1]

        last_x = self.cached_room_in_generation[batch_no][-2]
        last_y = self.cached_room_in_generation[batch_no][-1]

        ending_line_type = line_type(first_x, first_y, last_x, last_y)

        if ending_line_type == LineType.Diagonal:
            self.invalidate_room_end(logits, batch_no)

    
    def init_cache(self, input_ids):
        """
        Initializes the caches for generation state tracking.
        
        Converts boundary sequences to polygons and prepares tracking structures for room generation.
        
        :param input_ids: Input token IDs containing boundary information
        """
        boundary_door_seqs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        boundaries = [boundary_from_sequence(seq) for seq in boundary_door_seqs]

        self.cached_remaining_empty_spaces = [shapely.Polygon(boundary) for boundary in boundaries]
        self.cached_room_in_generation = [None] * input_ids.shape[0]
        for i in range(len(self.cached_room_in_generation)):
            self.cached_room_in_generation[i] = []

        self.cached_prev_token_to_gen = torch.ones(input_ids.shape[0], dtype=torch.int32) * TokenToGenerateType.FirstRoom.value

    
    def update_generated_room(self, input_ids, batch_no):
        """
        Updates the current room coordinates with a newly generated coordinate token.
        
        :param input_ids: Current input token IDs
        :param batch_no: Batch index
        """
        id = input_ids[batch_no, 0].item()
        coord = tokens.coord_from_token_id(id)

        self.cached_room_in_generation[batch_no].append(coord)


    def mask_x_direction(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
        """
        Masks valid X coordinates based on intersection geometry.
        
        Handles various geometry types (Point, LineString, MultiPoint, MultiLineString) to
        extract valid X coordinate ranges.
        
        :param intersection: Shapely geometry of intersection
        :param point: Reference point for filtering
        :type point: shapely.Point
        :param is_valid: Tensor to mark valid coordinate tokens
        :type is_valid: torch.Tensor
        """
        if intersection.is_empty:
            return
        
        if intersection.geom_type == "MultiLineString":
            intersection = linemerge(intersection)

        match intersection.geom_type:
            case "Point":
                self.mask_x_direction_point(intersection, point, is_valid)

            case "LineString":
                self.mask_x_direction_line(intersection, point, is_valid)

            case "MultiPoint":
                for p in intersection.geoms:
                    self.mask_x_direction_point(p, point, is_valid)

            case "MultiLineString":
                for line in intersection.geoms:
                    self.mask_x_direction_line(line, point, is_valid)

            case "GeometryCollection":
                lines = shapely.MultiLineString([line for line in intersection.geoms if line.geom_type == "LineString"])
                points = shapely.MultiPoint([p for p in intersection.geoms if p.geom_type == "Point"])

                assert len(lines.geoms) + len(points.geoms) == len(intersection.geoms) 

                self.mask_x_direction(lines, point, is_valid)
                self.mask_x_direction(points, point, is_valid)

            case _:
                raise Exception(f"Unexpected geometry type {intersection.geom_type}")


    def mask_x_direction_point(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
        """
        Extracts valid X coordinate from a point intersection.
        
        :param intersection: Point geometry
        :param point: Reference point
        :type point: shapely.Point
        :param is_valid: Tensor to mark valid token
        :type is_valid: torch.Tensor
        """
        if not intersection.intersects(point):
            return
        
        token_id = int(tokens.coord_token_id(intersection.x))
        is_valid[token_id] = True


    def mask_x_direction_line(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
        """
        Extracts valid X coordinate range from a line intersection.
        
        :param intersection: LineString geometry
        :param point: Reference point
        :type point: shapely.Point
        :param is_valid: Tensor to mark valid tokens
        :type is_valid: torch.Tensor
        """
        if not intersection.intersects(point):
            return
        
        (min_x, max_x) = self.geometry_x_bounds(intersection)

        min_token = tokens.coord_token_id(min_x)
        max_token = tokens.coord_token_id(max_x)

        is_valid[min_token:(max_token+1)] = True


    def mask_y_direction(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
        """
        Masks valid Y coordinates based on intersection geometry.
        
        Handles various geometry types (Point, LineString, MultiPoint, MultiLineString) to
        extract valid Y coordinate ranges.
        
        :param intersection: Shapely geometry of intersection
        :param point: Reference point for filtering
        :type point: shapely.Point
        :param is_valid: Tensor to mark valid coordinate tokens
        :type is_valid: torch.Tensor
        """
        if intersection.is_empty:
            return
        
        if intersection.geom_type == "MultiLineString":
            intersection = linemerge(intersection)

        match intersection.geom_type:
            case "Point":
                self.mask_y_direction_point(intersection, point, is_valid)

            case "LineString":
                self.mask_y_direction_line(intersection, point, is_valid)

            case "MultiPoint":
                for p in intersection.geoms:
                    self.mask_y_direction_point(p, point, is_valid)

            case "MultiLineString":
                for line in intersection.geoms:
                    self.mask_y_direction_line(line, point, is_valid)

            case "GeometryCollection":
                lines = shapely.MultiLineString([line for line in intersection.geoms if line.geom_type == "LineString"])
                points = shapely.MultiPoint([p for p in intersection.geoms if p.geom_type == "Point"])

                assert len(lines.geoms) + len(points.geoms) == len(intersection.geoms) 

                self.mask_y_direction(lines, point, is_valid)
                self.mask_y_direction(points, point, is_valid)

            case _:
                raise Exception(f"Unexpected geometry type {intersection.geom_type}")


    def mask_y_direction_point(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
        """
        Extracts valid Y coordinate from a point intersection.
        
        :param intersection: Point geometry
        :param point: Reference point
        :type point: shapely.Point
        :param is_valid: Tensor to mark valid token
        :type is_valid: torch.Tensor
        """
        if not intersection.intersects(point):
            return
        
        token_id = int(tokens.coord_token_id(intersection.y))
        is_valid[token_id] = True


    def mask_y_direction_line(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
        """
        Extracts valid Y coordinate range from a line intersection.
        
        :param intersection: LineString geometry
        :param point: Reference point
        :type point: shapely.Point
        :param is_valid: Tensor to mark valid tokens
        :type is_valid: torch.Tensor
        """
        if not intersection.intersects(point):
            return
        
        (min_y, max_y) = self.geometry_y_bounds(intersection)

        min_token = tokens.coord_token_id(min_y)
        max_token = tokens.coord_token_id(max_y)

        is_valid[min_token:(max_token+1)] = True


    def geometry_x_bounds(self, geom):
        """
        Extracts the X-axis bounds of a geometry.
        
        :param geom: Shapely geometry object
        :return: Tuple of (min_x, max_x) as integers
        :rtype: tuple[int, int]
        """
        (min_x, _, max_x, _) = geom.bounds

        min_x = int(min_x)
        max_x = int(max_x)

        return (min_x, max_x)


    def geometry_y_bounds(self, geom):
        """
        Extracts the Y-axis bounds of a geometry.
        
        :param geom: Shapely geometry object
        :return: Tuple of (min_y, max_y) as integers
        :rtype: tuple[int, int]
        """
        (_, min_y, _, max_y) = geom.bounds

        min_y = int(min_y)
        max_y = int(max_y)

        return (min_y, max_y)
    

    def get_is_valid_tensor_with_rooms_as_valid(self) -> torch.Tensor:
        """
        Creates a validity tensor with room type tokens marked as valid.
        
        :return: Boolean tensor with room tokens set to True
        :rtype: torch.Tensor
        """
        result = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        # Make room tokens valid
        result[tokens.MIN_ROOM_ID:(tokens.MAX_ROOM_ID+1)] = True

        return result
    

    def get_linestring_of_current_room(self, batch_no) -> shapely.LineString | shapely.Point:
        """
        Constructs a LineString from the current room's coordinates.
        
        Converts the flat coordinate list into corner pairs for geometric operations.
        
        :param batch_no: Batch index
        :return: LineString representation of room boundary
        :rtype: shapely.LineString
        """
        coordinates = self.cached_room_in_generation[batch_no]

        if len(coordinates) % 2 == 1:
            coordinates = coordinates[:-1]

        coordinates = torch.Tensor(coordinates)
        corners = coordinates.view(-1, 2)

        if corners.shape[0] == 1:
            return shapely.Point(corners)
        
        return shapely.LineString(corners)
    

    def finish_room_generation(self, batch_no):
        """
        Finalizes a completed room and updates available space.
        
        Constructs a polygon from room coordinates, validates it, and subtracts it from
        the available empty space for subsequent room generation.
        
        :param batch_no: Batch index
        """
        if len(self.cached_room_in_generation[batch_no]) == 0:
            return
        
        corners = torch.Tensor(self.cached_room_in_generation[batch_no])
        corners = corners.view(-1, 2)

        room = shapely.Polygon(corners)
        if not room.is_valid:
            print(shapely.validation.explain_validity(room))
            room = shapely.make_valid(room)
            self.validity_problems += 1

        self.cached_remaining_empty_spaces[batch_no] = self.cached_remaining_empty_spaces[batch_no].difference(room)

        self.cached_room_in_generation[batch_no] = []


    def _get_empty_space(self, batch_no) -> shapely.Polygon:
        """
        Retrieves the polygon containing the given point from the available empty space.
        
        For MultiPolygon spaces, finds the specific polygon containing the point.
        
        :param batch_no: Batch index
        :param point: Query point
        :type point: shapely.Point
        :return: Polygon containing the point
        :rtype: shapely.Polygon
        :raises Exception: If point is outside all empty space polygons
        """
        empty_space_polygon = self.cached_remaining_empty_spaces[batch_no]

        line = self.get_linestring_of_current_room(batch_no)

        if empty_space_polygon.geom_type == "MultiPolygon":
            for polygon in empty_space_polygon.geoms:
                if polygon.covers(line):
                    return polygon
                
            raise Exception("Point is outside empty space")
        
        return empty_space_polygon
