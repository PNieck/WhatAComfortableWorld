import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.tokens as tokens
from src.floor_plan import RoomType

from abc import ABC, abstractmethod


def gaussian1d(mu, sigma, device):
    """
    Creates a discrete non-normalized gaussian PDF in the range [MIN_COORD_ID, MAX_COORD_ID]

    Parameters:
    mu (float): center of the normal distribution
    sigma (float): variance of the normal distribution
    device (String): device of the returned tensor
    Returns:
    1D-tensor (float): tensor with shape [res] containing the evaluated PDF
    """
    mu = torch.as_tensor(mu).view(-1,1)
    x = torch.arange(tokens.MIN_COORD_ID, tokens.MAX_COORD_ID, device=device)
    return torch.exp(-0.5*((x-mu)/sigma)**2)


class BaseErgoLoss(ABC):
    def __init__(self) -> None:
        self.base_loss_fun = nn.CrossEntropyLoss(ignore_index=-100)

        self.sigma = 1.0


    def ergonomic_loss(self, floor_plan_ids: torch.Tensor):
        return self._ergonomic_loss(floor_plan_ids.unsqueeze(0).float())


    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.base_loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


    def _get_prediction(self, labels: torch.Tensor, logits: torch.Tensor):
        device = labels.device

        probs = F.softmax(logits,-1)
        pred_ind = torch.argmax(probs,-1)

        pred_is_coord = tokens.is_coord(pred_ind)
        label_is_prompt = self._is_prompt(labels)
        label_is_coord = tokens.is_coord(labels)

        is_valid = pred_is_coord & label_is_coord & (~label_is_prompt)

        coords_for_interp = pred_ind[is_valid]

        gauss_weights = gaussian1d(coords_for_interp, self.sigma, device=device)
        row_ind = torch.nonzero(is_valid).squeeze()
        col_ind = torch.arange(tokens.MIN_COORD_ID, tokens.MAX_COORD_ID, device=device)

        v = probs[row_ind, tokens.MIN_COORD_ID:tokens.MAX_COORD_ID] * gauss_weights

        interp_val = torch.sum(col_ind * v, 1)
        prob_sum = torch.sum(v, 1)
        interp_val = interp_val / prob_sum

        return interp_val, is_valid


    def _is_prompt(self, labels: torch.Tensor):
        device = labels.device     
        idx = torch.where(labels == tokens.DOOR_TOKEN_ID)[0]
        idx += 4

        indices = torch.arange(labels.size(0), device=device)

        return indices <= idx


    def _replace_gt_with_predicted_values(self, labels: torch.Tensor, pred_values: torch.Tensor, is_valid: torch.Tensor) -> torch.Tensor:
        valid_cnt = torch.sum(is_valid)

        labels = labels.float()

        # replace gt values with predicted values
        floor_plans = labels.repeat((valid_cnt.item(), 1))

        cols_ind = is_valid.nonzero(as_tuple=True)[0]
        rows_ind = torch.arange(valid_cnt)

        floor_plans[rows_ind, cols_ind] = pred_values

        return floor_plans


    def _ergonomic_loss(self, plan_ids: torch.Tensor):
        device = plan_ids.device

        entrance_loss = torch.mean(self._entrance_loss(plan_ids))
        kitchen_loss = torch.mean(self._kitchens_loss(plan_ids))
        bathrooms_loss = torch.mean(self._bathroom_loss(plan_ids))
        balconies_loss = torch.mean(self._balconies_loss(plan_ids))

        valid_losses = torch.zeros(1, device=device)
        ergo_loss = torch.zeros(1, device=device)

        if entrance_loss >= 0.0:
            ergo_loss += entrance_loss
            valid_losses += 1

        if kitchen_loss >= 0.0:
            ergo_loss += kitchen_loss
            valid_losses += 1

        if bathrooms_loss >= 0.0:
            ergo_loss += bathrooms_loss
            valid_losses += 1

        if balconies_loss >= 0.0:
            ergo_loss += balconies_loss
            valid_losses += 1

        if valid_losses > 0:
            return ergo_loss / (valid_losses * 10)
        
        return -torch.ones(1, device=device)


    @abstractmethod
    def _entrance_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        pass


    @abstractmethod
    def _kitchens_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        pass


    @abstractmethod
    def _bathroom_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        pass


    @abstractmethod
    def _balconies_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        pass


    def _find_rooms_indices(self, plan_ids: torch.Tensor, room_type: RoomType):
        return torch.where(plan_ids[0] == tokens.room_token_id(room_type.value))[0]
    

    def _room_coords(self, plan_ids, room_start_idx) -> torch.Tensor:
        room_coords_ids = self._room_sequences(plan_ids, room_start_idx)
        room_coords = tokens.coord_from_token_id(room_coords_ids)

        rows_cnt = room_coords.shape[0]

        return room_coords.view(rows_cnt, -1, 2)
    

    def _room_sequences(self, plan_ids, room_start_idx) -> torch.Tensor:
        rooms_idx_mask = tokens.is_room(plan_ids[0])
        rooms_idx_mask[0:(room_start_idx+1)] = False

        if not torch.any(rooms_idx_mask):
            # Handling padding (with -100 in labels)
            eof_idx = torch.where(plan_ids[0] == tokens.END_SEQ_TOKEN_ID)[0]
            assert len(eof_idx) == 1

            eof_idx = eof_idx[0]
            return plan_ids[:, room_start_idx+1:eof_idx]
        
        next_room_start = torch.where(rooms_idx_mask)[0][0]
        return plan_ids[:, room_start_idx+1:next_room_start]
