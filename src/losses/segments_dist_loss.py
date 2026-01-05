import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.tokens as tokens
from src.floor_plan import RoomType


def gaussian1d(mu,sigma,res,device='cpu'):
  """
  Creates a discrete non-normalized gaussian PDF in the range [0,res-1]

  Parameters:
    mu (float): center of the normal distribution
    sigma (float): variance of the normal distribution
    res (int): determines the range [0,res-1] in which the normal distribution is evaluated
    device (String): device of the returned tensor
  Returns:
    1D-tensor (float): tensor with shape [res] containing the evaluated PDF
  """
  mu = torch.as_tensor(mu).view(-1,1)
  x = torch.arange(tokens.MIN_COORD_ID, tokens.MAX_COORD_ID, device=device)
  return torch.exp(-0.5*((x-mu)/sigma)**2) #* 1/(s*np.sqrt(2*np.pi)) 


class SegmentDistLoss:
    def __init__(self, device):
        self.base_loss_fun = nn.CrossEntropyLoss(ignore_index=-100)

        self.sigma = 1.0
        self.res = 256

        self.beta = 10.0

        self.max_ergo_loss = torch.ones(1, device=device) * 30.0


    def update_max_ergo_loss(self, train_dataloader: DataLoader, device):
        self.max_ergo_loss = torch.zeros(1, device=device)
        
        with torch.no_grad():
            for batch in train_dataloader:
                labels: torch.Tensor = batch["labels"]
                if labels.device != device:
                    labels = labels.to(device)

                for i in range(labels.shape[0]):
                    batch_labels: torch.Tensor = labels[i]

                    ergo_loss = self._ergonomic_loss(batch_labels.unsqueeze(0).float())
                    self.max_ergo_loss = torch.max(self.max_ergo_loss, ergo_loss)

        self.max_ergo_loss = self.max_ergo_loss.to("cpu")


    def __call__(self, output, labels: torch.Tensor):
        logits: torch.Tensor = output.logits

        loss_sum = torch.zeros(1)

        for i in range(labels.size(0)):
            batch_logits = logits[i,:,:]
            batch_labels = labels[i,:]

            batch_logits = batch_logits.to("cpu")
            batch_labels = batch_labels.to("cpu")

            # TODO: change
            # floor_plan_ergo = self._ergonomic_loss(batch_labels.unsqueeze(0).float())
            # floor_plan_ergo = torch.min(floor_plan_ergo, self.max_ergo_loss)
            floor_plan_ergo = 0.5
            weight = 1.0 - floor_plan_ergo / self.max_ergo_loss

            std_loss = self.cc_loss(batch_logits.unsqueeze(0), batch_labels.unsqueeze(0))

            if weight < 1.0:
                inter_val, is_valid = self._get_prediction(batch_labels, batch_logits)
                plan_ids = self._replace_gt_with_predicted_values(batch_labels, inter_val, is_valid)

                if plan_ids.numel() == 0:
                    loss = std_loss
                else:
                    ergo_loss = self._ergonomic_loss(plan_ids)
                    loss = weight * std_loss + (1.0 - weight) * ergo_loss
            else:
                loss = std_loss

            loss_sum += loss

        return loss_sum
    

    def ergonomic_loss(self, floor_plan_ids: torch.Tensor):
        return self._ergonomic_loss(floor_plan_ids.unsqueeze(0).float())


    def cc_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.base_loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


    def _get_prediction(self, labels: torch.Tensor, logits: torch.Tensor):
        device = labels.device

        probs = F.softmax(logits,-1)
        pred_ind = torch.argmax(probs,-1)
        #pred_ind = pred_ind.float()

        pred_is_coord = tokens.is_coord(pred_ind)
        label_is_prompt = self._is_prompt(labels)
        label_is_coord = tokens.is_coord(labels)

        is_valid = pred_is_coord & label_is_coord & (~label_is_prompt)

        coords_for_interp = pred_ind[is_valid]

        gauss_weights = gaussian1d(coords_for_interp, self.sigma, self.res, device=device)
        row_ind = torch.nonzero(is_valid).squeeze()
        col_ind = torch.arange(tokens.MIN_COORD_ID, tokens.MAX_COORD_ID, device=device)
        # col_ind = torch.arange(0, self.res, device=device)

        v = probs[row_ind, tokens.MIN_COORD_ID:tokens.MAX_COORD_ID] * gauss_weights

        interp_val = torch.sum(col_ind * v, 1)
        prob_sum = torch.sum(v, 1)
        interp_val = interp_val / prob_sum

        #pred_ind[is_valid] = interp_val

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
            return ergo_loss / valid_losses
        
        return -torch.ones(1, device=device)


    def _entrance_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        device = plan_ids.device

        entrances_idx = self._find_rooms_indices(plan_ids, RoomType.Entrance)
        if len(entrances_idx) == 0:
            return -torch.ones(1, device=device)
        
        door_idx = torch.where(plan_ids[0] == tokens.DOOR_TOKEN_ID)[0]

        door_x1 = tokens.coord_from_token_id(plan_ids[0][door_idx + 1])
        door_y1 = tokens.coord_from_token_id(plan_ids[0][door_idx + 2])
        door_x2 = tokens.coord_from_token_id(plan_ids[0][door_idx + 3])
        door_y2 = tokens.coord_from_token_id(plan_ids[0][door_idx + 4])

        door_points = torch.stack([door_x1, door_y1, door_x2, door_y2]).view(2, 2)

        door_starts = door_points.unsqueeze(0)
        door_ends = torch.roll(door_starts, -1, 1)

        losses = torch.empty((len(entrances_idx), plan_ids.shape[0]), device=device)

        for i, entrance_id in enumerate(entrances_idx):
            entrance_points = self._room_coords(plan_ids, entrance_id)

            entrance_starts = entrance_points
            entrance_ends = torch.roll(entrance_points, -1, 1)

            distances = self.segment_segment_distance_double_batch(
                door_starts, door_ends,
                entrance_starts, entrance_ends
            )

            distances = distances.view(entrance_points.shape[0], -1)
            loss = torch.sum(distances * F.softmin(distances * self.beta, -1), -1)

            losses[i, :] = loss

        return losses.mean(0)


    def _kitchens_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        return self._calculate_loss(plan_ids, RoomType.Kitchen, [RoomType.Entrance, RoomType.DiningRoom])


    def _bathroom_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        neighbors_types = [RoomType.Entrance, RoomType.LivingRoom, RoomType.MasterRoom, RoomType.SecondRoom]

        return self._calculate_loss(plan_ids, RoomType.Bathroom, neighbors_types)


    def _balconies_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        balconies_idx = self._find_rooms_indices(plan_ids, RoomType.Balcony)
        if len(balconies_idx) == 0:
            return -torch.ones(1)
        
        neighbors_types = [
            RoomType.LivingRoom,
            RoomType.StudyRoom,
            RoomType.Kitchen,
            RoomType.MasterRoom,
            RoomType.SecondRoom
        ]

        neighbors_idx = [None] * len(neighbors_types)

        for i, neighbor_type in enumerate(neighbors_types):
            neighbors_idx[i] = self._find_rooms_indices(plan_ids, neighbor_type)

        neighbors_idx = torch.concat(neighbors_idx)
        if len(neighbors_idx) == 0:
            return -torch.ones(1)
        
        interp_values_cnt = plan_ids.shape[0]
        losses = torch.empty((len(balconies_idx), len(neighbors_idx), interp_values_cnt))

        for i, balcony_id in enumerate(balconies_idx):
            balcony_coords = self._room_coords(plan_ids, balcony_id)

            balcony_starts = balcony_coords
            balcony_ends = torch.roll(balcony_coords, -1, 1)

            for j, neighbor_id in enumerate(neighbors_idx):
                neighbor_coords = self._room_coords(plan_ids, neighbor_id)

                neighbor_starts = neighbor_coords
                neighbor_ends = torch.roll(neighbor_coords, -1, 1)

                distances = self.segment_segment_distance_double_batch(
                    balcony_starts, balcony_ends,
                    neighbor_starts, neighbor_ends
                )

                distances = distances.view(interp_values_cnt, -1)
                loss = torch.sum(distances * F.softmin(distances * self.beta, -1), -1)

                losses[i, j, :] = loss

        losses = torch.sum(losses * F.softmin(losses * self.beta, 0), 0)
        losses = torch.sum(losses * F.softmin(losses * self.beta, 0), 0)

        return losses



    def _calculate_loss(self, plan_ids: torch.Tensor, main_type: RoomType, neighbors_types: list[RoomType]) -> torch.Tensor:
        device = plan_ids.device
        
        main_rooms_idx = self._find_rooms_indices(plan_ids, main_type)
        if len(main_rooms_idx) == 0:
            return -torch.ones(1, device=device)
        
        neighbors_idx = [None] * len(neighbors_types)

        for i, neighbor_type in enumerate(neighbors_types):
            neighbors_idx[i] = self._find_rooms_indices(plan_ids, neighbor_type)

        neighbors_idx = torch.concat(neighbors_idx)
        if len(neighbors_idx) == 0:
            return -torch.ones(1, device=device)
        
        interp_values_cnt = plan_ids.shape[0]
        losses = torch.empty((len(main_rooms_idx), len(neighbors_idx), interp_values_cnt), device=device)

        for i, main_room_id in enumerate(main_rooms_idx):
            main_room_coords = self._room_coords(plan_ids, main_room_id)

            main_rooms_starts = main_room_coords
            main_rooms_ends = torch.roll(main_room_coords, -1, 1)

            for j, neighbor_id in enumerate(neighbors_idx):
                neighbor_coords = self._room_coords(plan_ids, neighbor_id)

                neighbor_starts = neighbor_coords
                neighbor_ends = torch.roll(neighbor_coords, -1, 1)

                distances = self.segment_segment_distance_double_batch(
                    main_rooms_starts, main_rooms_ends,
                    neighbor_starts, neighbor_ends
                )

                distances = distances.view(interp_values_cnt, -1)
                loss = torch.sum(distances * F.softmin(distances * self.beta, -1), -1)

                losses[i, j, :] = loss

        losses = torch.sum(losses * F.softmin(losses * self.beta, 0), 0)
        losses = losses.mean(0)

        return losses
    

    def segment_segment_distance_double_batch(
        self, p, p2, q, q2, eps=1e-9
    ):
        """
        Double-batch all-pairs segment–segment distance.

        p, p2, q, q2: (A, B, D)
        returns: (A, B, B)
        """

        # Direction vectors
        u = p2 - p          # (A, B, D)
        v = q2 - q          # (A, B, D)

        # Expand for pairwise computation
        p = p[:, :, None, :]    # (A, B, 1, D)
        q = q[:, None, :, :]    # (A, 1, B, D)

        u = u[:, :, None, :]    # (A, B, 1, D)
        v = v[:, None, :, :]    # (A, 1, B, D)

        w = p - q               # (A, B, B, D)

        # Dot products
        a = (u * u).sum(dim=-1)     # (A, B, 1)
        b = (u * v).sum(dim=-1)     # (A, B, B)
        c = (v * v).sum(dim=-1)     # (A, 1, B)
        d = (u * w).sum(dim=-1)     # (A, B, B)
        e = (v * w).sum(dim=-1)     # (A, B, B)

        # Solve for closest points
        D = a * c - b * b
        D = D + eps

        t = (b * e - c * d) / D
        s = (a * e - b * d) / D

        t = t.clamp(0.0, 1.0)
        s = s.clamp(0.0, 1.0)

        # Closest points
        closest_p = p + t[..., None] * u    # (A, B, B, D)
        closest_q = q + s[..., None] * v    # (A, B, B, D)

        diff = closest_p - closest_q
        dist2 = (diff * diff).sum(dim=-1)   # (A, B, B)

        return dist2


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
