from enum import Enum
from typing import List

import torch
import shapely
from shapely.ops import linemerge
from transformers import GPT2Config, GPT2LMHeadModel

import src.tokens as tokens
from src.floor_plan_tokenizer import FloorPlanTokenizer
from src.floor_plan import RoomType
from src.sequence.from_sequence import boundary_from_sequence
from src.geom_utils import LineType, line_type


def get_gpt2_config(config) -> GPT2Config:
    return GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config["max_seq_len"],
        n_ctx=config["max_seq_len"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        bos_token_id=tokens.START_SEQ_TOKEN_ID,
        eos_token_id=tokens.END_SEQ_TOKEN_ID,
    )


def get_gpt2(config):
    config = get_gpt2_config(config)

    model = CustomGPT2(config)

    return model


class TokenToGenerateType(Enum):
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
    def __init__(self, config: GPT2Config):
        self.max_seq_len = config.n_embd
        self.use_masked_inference = False

        super().__init__(config)

        self.cached_remaining_empty_spaces: List[shapely.Polygon] = None
        self.cached_room_in_generation: List[List[int]] = None

        self.cached_prev_token_to_gen: torch.Tensor = None

        self.tokenizer = FloorPlanTokenizer()
    
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs
    ):
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
        assert use_cache == True, "Not using cache is not supported"
        assert input_ids is not None

        # Initial masking
        logits[:, :, tokens.START_SEQ_TOKEN_ID] = -torch.inf
        logits[:, :, tokens.UNK_TOKEN_ID] = -torch.inf
        logits[:, :, tokens.PAD_TOKEN_ID] = -torch.inf
        logits[:, :, tokens.BOUNDARY_TOKEN_ID] = -torch.inf
        logits[:, :, tokens.DOOR_TOKEN_ID] = -torch.inf

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
                    self.mask_for_first_y_in_room(input_ids, logits, batch_no)

                case TokenToGenerateType.SecondXInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_second_x_in_room(logits, batch_no)

                case TokenToGenerateType.SecondYInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_second_y_in_room(logits, batch_no)

                case TokenToGenerateType.ThirdXInRoom:
                    self.update_generated_room(input_ids, batch_no)
                    self.mask_for_third_x_in_room(logits, batch_no)


            self.cached_prev_token_to_gen[batch_no] = token_to_generate.value
            

        return logits


    def get_token_to_generate_type(self, input_ids, batch_no) -> TokenToGenerateType:
        if self.cached_remaining_empty_spaces is None:
            return TokenToGenerateType.FirstRoom
        
        id = input_ids[batch_no, 0]
        if id >= tokens.MIN_ROOM_ID and id <= tokens.MAX_ROOM_ID:
            return TokenToGenerateType.FirstXInRoom
        
        result = TokenToGenerateType(self.cached_prev_token_to_gen[batch_no].item() + 1)
        if result.value >= TokenToGenerateType.NthXInRoom.value:
            if self.cached_remaining_empty_spaces[batch_no].is_empty:
                result = TokenToGenerateType.EndToken
            
            elif result.value == TokenToGenerateType.EndToken.value:
                result = TokenToGenerateType.NthXInRoom
            
        return result


    def mask_for_first_room(self, logits: torch.Tensor):
        logits[:, :, tokens.MIN_COORD_ID:tokens.MAX_COORD_ID] = -torch.inf
        logits[:, :, tokens.END_SEQ_TOKEN_ID] = -torch.inf
    

    def mask_for_first_x_in_room(self, logits, batch_no):
        polygon = self.cached_remaining_empty_spaces[batch_no]
        (min_x, max_x) = self.geometry_x_bounds(polygon)

        is_valid = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        # TODO: faster algorithm
        for x in range(min_x, max_x+1):
            line = shapely.LineString([(x, 0), (x, 256)])

            if polygon.intersects(line):
                is_valid[tokens.coord_token_id(x)] = True

        logits[batch_no, :, ~is_valid] = -torch.inf

    
    def mask_for_first_y_in_room(self, input_ids, logits, batch_no):
        polygon = self.cached_remaining_empty_spaces[batch_no]
        (_, min_y, _, max_y) = polygon.bounds

        x = tokens.coord_from_token_id(input_ids[batch_no, 0])

        line = shapely.LineString([(x, min_y), (x, max_y)])
        intersection = polygon.intersection(line)
        if intersection.geom_type == "MultiLineString":
            intersection = linemerge(intersection)

        is_valid = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        assert not intersection.is_empty

        if intersection.geom_type == "LineString":
            (min_y, max_y) = self.geometry_y_bounds(intersection)

            assert len(intersection.coords) == 2

            min_token = tokens.coord_token_id(min_y)
            max_token = tokens.coord_token_id(max_y)

            is_valid[min_token:(max_token+1)] = True

        elif intersection.geom_type == "MultiLineString":
            for line in intersection.geoms:
                (min_y, max_y) = self.geometry_y_bounds(line)
                
                assert len(intersection.coords) == 2

                min_token = tokens.coord_token_id(min_y)
                max_token = tokens.coord_token_id(max_y)

                is_valid[min_token:(max_token+1)] = True

        else:
            assert False, "Unknown geometry type"

        logits[batch_no, :, ~is_valid] = -torch.inf


    def mask_for_second_x_in_room(self, logits, batch_no):
        polygon = self.cached_remaining_empty_spaces[batch_no]
        (min_x, max_x) = self.geometry_x_bounds(polygon)

        first_x = self.cached_room_in_generation[batch_no][0]
        first_y = self.cached_room_in_generation[batch_no][1]

        first_point = shapely.Point(first_x, first_y)

        is_valid = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        if first_x != min_x:
            line = shapely.LineString([(min_x, first_y), (first_x, first_y)])
            inter = polygon.intersection(line)
            self.mask_x_direction(inter, first_point, is_valid)

        if first_x != max_x:
            line = shapely.LineString([(first_x, first_y), (max_x, first_y)])
            inter = polygon.intersection(line)
            self.mask_x_direction(inter, first_point, is_valid)
                
        logits[batch_no, :, ~is_valid] = -torch.inf


    def mask_for_second_y_in_room(self, logits, batch_no):
        first_x = self.cached_room_in_generation[batch_no][0]
        first_y = self.cached_room_in_generation[batch_no][1]

        last_x = self.cached_room_in_generation[batch_no][-1]
        last_y = first_y

        last_y_token_id = tokens.coord_token_id(last_y)

        is_valid = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        if last_x != first_x:
            is_valid[last_y_token_id] = True

        else:
            polygon = self.cached_remaining_empty_spaces[batch_no]
            (min_y, max_y) = self.geometry_y_bounds(polygon)

            first_point = shapely.Point(first_x, first_y)

            if first_y != min_y:
                line = shapely.LineString([(last_x, min_y), (last_x, first_y)])
                inter = polygon.intersection(line)
                self.mask_y_direction(inter, first_point, is_valid)

            if first_x != max_y:
                line = shapely.LineString([(last_x, first_y), (last_x, max_y)])
                inter = polygon.intersection(line)
                self.mask_y_direction(inter, first_point, is_valid)

            is_valid[last_y_token_id] = False     

        logits[batch_no, :, ~is_valid] = -torch.inf


    def mask_for_third_x_in_room(self, logits, batch_no):
        first_x = self.cached_room_in_generation[batch_no][0]
        first_y = self.cached_room_in_generation[batch_no][1]

        second_x = self.cached_room_in_generation[batch_no][2]
        second_y = self.cached_room_in_generation[batch_no][3]

        last_line_type = line_type(first_x, first_y, second_x, second_y)
        assert last_line_type != LineType.Diagonal

        is_valid = torch.zeros(len(self.tokenizer), dtype=torch.bool)

        polygon = self.cached_remaining_empty_spaces[batch_no]
        (min_x, max_x) = self.geometry_x_bounds(polygon)

        last_point = shapely.Point(second_x, second_y)

        if last_line_type == LineType.Vertical:
            

            if second_x != min_x:
                line = shapely.LineString([(min_x, second_y), (second_x, second_y)])
                inter = polygon.intersection(line)
                self.mask_x_direction(inter, last_point, is_valid)

            if second_x != max_x:
                line = shapely.LineString([(second_x, second_y), (max_x, second_y)])
                inter = polygon.intersection(line)
                self.mask_x_direction(inter, last_point, is_valid)

        else:
            # Line is Horizontal

            if first_x < second_x:
                line = shapely.LineString([(second_x, second_y), (max_x, second_y)])
                inter = polygon.intersection(line)
                self.mask_x_direction(inter, last_point, is_valid)

            else:
                line = shapely.LineString([(min_x, second_y), (second_x, second_y)])
                inter = polygon.intersection(line)
                self.mask_x_direction(inter, last_point, is_valid)

        logits[batch_no, :, ~is_valid] = -torch.inf
            

    
    def init_cache(self, input_ids):
        boundary_door_seqs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        boundaries = [boundary_from_sequence(seq) for seq in boundary_door_seqs]

        self.cached_remaining_empty_spaces = [shapely.Polygon(boundary) for boundary in boundaries]
        self.cached_room_in_generation = [None] * input_ids.shape[0]
        for i in range(len(self.cached_room_in_generation)):
            self.cached_room_in_generation[i] = []

        self.cached_prev_token_to_gen = torch.ones(input_ids.shape[0], dtype=torch.int32) * TokenToGenerateType.FirstRoom.value

    
    def update_generated_room(self, input_ids, batch_no):
        id = input_ids[batch_no, 0].item()
        coord = tokens.coord_from_token_id(id)

        self.cached_room_in_generation[batch_no].append(coord)


    def mask_x_direction(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
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
        if not intersection.intersects(point):
            return
        
        token_id = int(tokens.coord_token_id(intersection.x))
        is_valid[token_id] = True


    def mask_x_direction_line(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
        if not intersection.intersects(point):
            return
        
        (min_x, max_x) = self.geometry_x_bounds(intersection)

        min_token = tokens.coord_token_id(min_x)
        max_token = tokens.coord_token_id(max_x)

        is_valid[min_token:(max_token+1)] = True


    def mask_y_direction(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
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
        if not intersection.intersects(point):
            return
        
        token_id = int(tokens.coord_token_id(intersection.y))
        is_valid[token_id] = True


    def mask_y_direction_line(self, intersection, point: shapely.Point, is_valid: torch.Tensor):
        if not intersection.intersects(point):
            return
        
        (min_y, max_y) = self.geometry_y_bounds(intersection)

        min_token = tokens.coord_token_id(min_y)
        max_token = tokens.coord_token_id(max_y)

        is_valid[min_token:(max_token+1)] = True


    def geometry_x_bounds(self, geom):
        (min_x, _, max_x, _) = geom.bounds

        min_x = int(min_x)
        max_x = int(max_x)

        return (min_x, max_x)
    
    def geometry_y_bounds(self, geom):
        (_, min_y, _, max_y) = geom.bounds

        min_y = int(min_y)
        max_y = int(max_y)

        return (min_y, max_y)
