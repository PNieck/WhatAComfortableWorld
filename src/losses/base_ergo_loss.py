"""Base class for ergonomic loss functions in floor plan generation.

Provides abstract base class with shared methods for computing ergonomic constraints
on floor plans, including room adjacency and distance metrics.
"""

import torch
import torch.nn  as nn
import torch.nn.functional as F

import src.tokens as tokens
from src.floor_plan import RoomType

from abc import ABC, abstractmethod


def gaussian1d(mu, sigma, device):
    """Create a discrete non-normalized 1D Gaussian PDF over coordinate range.
    
    :param mu: Center of the normal distribution
    :param sigma: Standard deviation of the distribution
    :param device: PyTorch device for the returned tensor
    :return: 1D tensor containing evaluated Gaussian PDF over coordinate range
    """

    mu = torch.as_tensor(mu).view(-1,1)
    x = torch.arange(tokens.MIN_COORD_ID, tokens.MAX_COORD_ID, device=device)
    return torch.exp(-0.5*((x-mu)/sigma)**2)


class BaseErgoLoss(ABC):
    """Abstract base class for ergonomic loss calculations in floor plan generation.
    
    Provides methods for computing ergonomic constraints on floor plans including
    cross-entropy loss and various ergonomic losses for different room type relationships.
    """
    
    def __init__(self, scaling=1.0) -> None:
        """Initialize BaseErgoLoss with cross-entropy loss and Gaussian sigma."""
        self.base_loss_fun = nn.CrossEntropyLoss(ignore_index=-100)

        self.sigma = 1.0
        self.scaling_factor = scaling


    def ergonomic_loss(self, floor_plan_ids: torch.Tensor):
        """
        Computes the ergonomic loss for a floor plan.
        
        :param floor_plan_ids: Floor plan token IDs
        :type floor_plan_ids: torch.Tensor
        :return: Ergonomic loss value
        :rtype: torch.Tensor
        """
        return self._ergonomic_loss(floor_plan_ids.unsqueeze(0).float())


    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Computes the cross-entropy loss between predictions and ground truth labels.
        
        :param logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        :type logits: torch.Tensor
        :param labels: Ground truth token IDs of shape (batch_size, seq_len)
        :type labels: torch.Tensor
        :return: Scalar cross-entropy loss
        :rtype: torch.Tensor
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.base_loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


    def _get_prediction(self, labels: torch.Tensor, logits: torch.Tensor):
        """
        Extracts and interpolates coordinate predictions from logits using gaussian weighting.
        
        Identifies valid coordinate predictions, applies gaussian weighting for interpolation,
        and returns interpolated values. Valid coordinates are ones where the network correctly
        predicted room coordinate.
        
        :param labels: Ground truth labels tensor
        :type labels: torch.Tensor
        :param logits: Model output logits
        :type logits: torch.Tensor
        :return: Tuple of (interpolated coordinate values, validity mask)
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

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
        """
        Determines which token positions are part of the prompt (input) sequence.
        
        Prompt contains floor plan boundary and front door sequence
        
        :param labels: Token label tensor
        :type labels: torch.Tensor
        :return: Boolean mask indicating prompt positions
        :rtype: torch.Tensor
        """
        device = labels.device     
        idx = torch.where(labels == tokens.DOOR_TOKEN_ID)[0]
        idx += 4

        indices = torch.arange(labels.size(0), device=device)

        return indices <= idx


    def _replace_gt_with_predicted_values(self, labels: torch.Tensor, pred_values: torch.Tensor, is_valid: torch.Tensor) -> torch.Tensor:
        """
        Replaces ground truth coordinate values with predicted values at valid positions.
        
        Creates copies of the ground truth labels and substitutes predicted values at positions
        marked as valid.
        
        :param labels: Ground truth labels tensor
        :type labels: torch.Tensor
        :param pred_values: Predicted coordinate values
        :type pred_values: torch.Tensor
        :param is_valid: Boolean mask indicating valid positions for replacement
        :type is_valid: torch.Tensor
        :return: Modified floor plan tensor with replaced values
        :rtype: torch.Tensor
        """

        valid_cnt = torch.sum(is_valid)

        labels = labels.float()

        # replace gt values with predicted values
        floor_plans = labels.repeat((valid_cnt.item(), 1))

        cols_ind = is_valid.nonzero(as_tuple=True)[0]
        rows_ind = torch.arange(valid_cnt)

        floor_plans[rows_ind, cols_ind] = pred_values

        return floor_plans


    def _ergonomic_loss(self, plan_ids: torch.Tensor):
        """
        Computes the combined ergonomic loss for all constraints.
        
        Aggregates individual losses for entrance, kitchen, bathroom, and balcony constraints.
        Returns a weighted average of valid losses, or -1 if no valid losses are found.
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :return: Combined ergonomic loss or -1 tensor if no valid losses
        :rtype: torch.Tensor
        """

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
            return ergo_loss / (valid_losses * self.scaling_factor)
        
        return -torch.ones(1, device=device)


    @abstractmethod
    def _entrance_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the entrance ergonomic loss for a floor plan.
        
        Must be implemented by subclasses to define specific entrance ergonomic constraints.
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :return: Entrance ergonomic loss
        :rtype: torch.Tensor
        """

        pass


    @abstractmethod
    def _kitchens_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the kitchen ergonomic loss for a floor plan.
        
        Must be implemented by subclasses to define specific kitchen ergonomic constraints.
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :return: Kitchen ergonomic loss
        :rtype: torch.Tensor
        """

        pass


    @abstractmethod
    def _bathroom_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the bathroom ergonomic loss for a floor plan.
        
        Must be implemented by subclasses to define specific bathroom ergonomic constraints.
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :return: Bathroom constraint loss
        :rtype: torch.Tensor
        """

        pass


    @abstractmethod
    def _balconies_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the balcony ergonomic loss for a floor plan.
        
        Must be implemented by subclasses to define specific balcony ergonomic constraints.
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :return: Balcony ergonomic loss
        :rtype: torch.Tensor
        """
        pass


    def _find_rooms_indices(self, plan_ids: torch.Tensor, room_type: RoomType):
        """
        Finds starts tokens of specified room types in a floor plan
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :param room_type: Type of a room to find start tokens
        :type room_type: RoomType
        :return: Indices of start tokens
        :rtype: torch.Tensor
        """

        return torch.where(plan_ids[0] == tokens.room_token_id(room_type.value))[0]
    

    def _room_coords(self, plan_ids, room_start_idx) -> torch.Tensor:
        """
        Returns vertices of a rooms, which starts for specified room start token index
        
        :param plan_ids: Floor plan token IDs
        :param room_start_idx: Index of a room start token
        :return: Vertices of a room
        :rtype: torch.Tensor
        """

        room_coords_ids = self._room_sequences(plan_ids, room_start_idx)
        room_coords = tokens.coord_from_token_id(room_coords_ids)

        rows_cnt = room_coords.shape[0]

        return room_coords.view(rows_cnt, -1, 2)
    

    def _room_sequences(self, plan_ids, room_start_idx) -> torch.Tensor:
        """
        Returns parts of a floor plan sequence, which describes specified room shape
        
        :param plan_ids: Floor plan token IDs
        :param room_start_idx: Index of a room start token
        :return: Parts of a sequence with room polygon shape
        :rtype: Tensor
        """
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
