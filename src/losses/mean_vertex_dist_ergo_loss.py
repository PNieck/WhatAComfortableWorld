import torch
import torch.nn.functional as F

import src.tokens as tokens
from src.floor_plan import RoomType

from .base_ergo_loss import BaseErgoLoss


class MeanVertexDistErgoLoss(BaseErgoLoss):
    """
    Computes a weighted combination of standard cross-entropy loss and ergonomic loss.
    
    This loss function enforces neighborhood relationships between rooms in floor plans by
    calculating distances between related room types (e.g., kitchens near entrances, bathrooms
    near living areas). Uses adaptive weighting based on ergonomic score.

    Neighborhood is calculated based on distances between mean vertex value of room's polygons.
    Approximation of distances between room's centroids.
    """

    def __init__(self):
        super().__init__(scaling=10.0)

        self.beta = 10.0

        self.max_ergo_loss = torch.ones(1, device="cpu") * 10.0
        self.min_ergo_loss = torch.ones(1, device="cpu") * 5.0


    def __call__(self, output, labels: torch.Tensor):
        logits: torch.Tensor = output.logits

        loss_sum = torch.zeros(1)

        for i in range(labels.size(0)):
            batch_logits = logits[i,:,:]
            batch_labels = labels[i,:]

            batch_logits = batch_logits.to("cpu")
            batch_labels = batch_labels.to("cpu")

            floor_plan_ergo = self._ergonomic_loss(batch_labels.unsqueeze(0).float())

            if floor_plan_ergo < self.min_ergo_loss:
                weight = 1.0
            else:
                tmp = self.max_ergo_loss - self.min_ergo_loss
                floor_plan_ergo = torch.min(floor_plan_ergo - self.min_ergo_loss, tmp)
                weight = 1.0 - floor_plan_ergo / self.max_ergo_loss

            std_loss = self.cross_entropy_loss(batch_logits.unsqueeze(0), batch_labels.unsqueeze(0))

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

        return loss_sum.squeeze()


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

        door_mean = door_points.mean(0)

        losses = torch.empty((len(entrances_idx), plan_ids.shape[0]), device=device)

        for i, entrance_id in enumerate(entrances_idx):
            entrance_points = self._room_coords(plan_ids, entrance_id)

            entrance_mean = entrance_points.mean(1)
            diff = entrance_mean - door_mean

            loss = torch.linalg.vector_norm(diff, dim=1)

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
            balcony_mean = balcony_coords.mean(1)

            for j, neighbor_id in enumerate(neighbors_idx):
                neighbor_coords = self._room_coords(plan_ids, neighbor_id)
                neighbor_room_mean = neighbor_coords.mean(1)

                diff = balcony_mean - neighbor_room_mean
                loss = torch.linalg.vector_norm(diff, dim=1)

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
            main_room_mean = main_room_coords.mean(1)

            for j, neighbor_id in enumerate(neighbors_idx):
                neighbor_coords = self._room_coords(plan_ids, neighbor_id)
                neighbor_room_mean = neighbor_coords.mean(1)

                diff = main_room_mean - neighbor_room_mean
                loss = torch.linalg.vector_norm(diff, dim=1)

                losses[i, j, :] = loss

        losses = torch.sum(losses * F.softmin(losses * self.beta, 0), 0)
        losses = losses.mean(0)

        return losses
