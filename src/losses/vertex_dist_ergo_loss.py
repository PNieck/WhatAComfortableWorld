import torch
import torch.nn.functional as F

import src.tokens as tokens
from src.floor_plan import RoomType
from .base_ergo_loss import BaseErgoLoss


class VertexDistancesErgoLoss(BaseErgoLoss):
    """
    Computes a weighted combination of standard cross-entropy loss and ergonomic loss.
    
    This loss function enforces neighborhood relationships between rooms in floor plans by
    calculating distances between related room types (e.g., kitchens near entrances, bathrooms
    near living areas). Uses adaptive weighting based on ergonomic score.

    Neighborhood is calculated based on pair-wise distances between room's polygons vertices.
    """
    
    def __init__(self):
        """
        Initializes the NeighborhoodLoss with default parameters.
        
        Sets up the beta parameter for softmin temperature and the maximum ergonomic loss threshold.
        """
        super().__init__()

        self.beta = 10.0

        self.max_ergo_loss = torch.ones(1, device="cpu") * 30.0


    def __call__(self, output, labels: torch.Tensor):
        """
        Computes the combined loss for all batch items.
        
        For each item in the batch, calculates the ergonomic score and uses it to adaptively
        weight between standard cross-entropy loss and ergonomic loss. Higher ergonomic scores
        result in more emphasis on the cross-entropy loss.
        
        :param output: Model output containing logits
        :param labels: Ground truth token IDs of shape (batch_size, seq_len)
        :return: Summed combined loss across the batch
        :rtype: torch.Tensor
        """
        logits: torch.Tensor = output.logits

        loss_sum = torch.zeros(1)

        for i in range(labels.size(0)):
            batch_logits = logits[i,:,:]
            batch_labels = labels[i,:]

            batch_logits = batch_logits.to("cpu")
            batch_labels = batch_labels.to("cpu")

            floor_plan_ergo = self._ergonomic_loss(batch_labels.unsqueeze(0).float())
            floor_plan_ergo = torch.min(floor_plan_ergo, self.max_ergo_loss)
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
    

    def ergonomic_loss_output(self, output, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes only the ergonomic loss component using predicted values.
        
        Extracts predictions from logits, replaces ground truth with interpolated predictions,
        and calculates ergonomic loss for the modified floor plans.
        
        :param output: Model output containing logits
        :param labels: Ground truth token IDs of shape (batch_size, seq_len)
        :return: Summed ergonomic loss across the batch
        :rtype: torch.Tensor
        """
        logits: torch.Tensor = output.logits

        loss_sum = torch.zeros(1)

        for i in range(labels.size(0)):
            batch_logits = logits[i,:,:]
            batch_labels = labels[i,:]

            batch_logits = batch_logits.to("cpu")
            batch_labels = batch_labels.to("cpu")

            inter_val, is_valid = self._get_prediction(batch_labels, batch_logits)
            plan_ids = self._replace_gt_with_predicted_values(batch_labels, inter_val, is_valid)

            if plan_ids.numel() == 0:
                continue

            ergo_loss = self._ergonomic_loss(plan_ids)
            loss_sum += ergo_loss

        return loss_sum.squeeze()
    

    def std_loss(self, output, labels: torch.Tensor):
        """
        Computes only the standard cross-entropy loss without ergonomic weighting.
        
        :param output: Model output containing logits
        :param labels: Ground truth token IDs of shape (batch_size, seq_len)
        :return: Summed cross-entropy loss across the batch
        :rtype: torch.Tensor
        """
        logits: torch.Tensor = output.logits

        loss_sum = torch.zeros(1)

        for i in range(labels.size(0)):
            batch_logits = logits[i,:,:]
            batch_labels = labels[i,:]

            batch_logits = batch_logits.to("cpu")
            batch_labels = batch_labels.to("cpu")

            std_loss = self.cross_entropy_loss(batch_logits.unsqueeze(0), batch_labels.unsqueeze(0))
            
            loss_sum += std_loss

        return loss_sum.squeeze()


    def _entrance_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the entrance ergonomic constraint loss.
        
        Calculates the minimum distance from the door to all entrance rooms using softmin weighting.
        Penalizes when entrances are far from the door position.
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :return: Entrance constraint loss
        :rtype: torch.Tensor
        """
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

        losses = torch.empty((len(entrances_idx), plan_ids.shape[0]), device=device)

        for i, entrance_id in enumerate(entrances_idx):
            entrance_points = self._room_coords(plan_ids, entrance_id)

            distances = torch.cdist(door_points, entrance_points).view(entrance_points.shape[0], -1)
            loss = torch.sum(distances * F.softmin(distances * self.beta, -1), -1)

            losses[i, :] = loss

        return losses.mean(0)


    def _kitchens_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the kitchen ergonomic constraint loss.
        
        Enforces that kitchens are near entrances and dining rooms.
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :return: Kitchen constraint loss
        :rtype: torch.Tensor
        """
        return self._calculate_loss(plan_ids, RoomType.Kitchen, [RoomType.Entrance, RoomType.DiningRoom])


    def _bathroom_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the bathroom ergonomic constraint loss.
        
        Enforces that bathrooms are near entrances, living rooms, bedrooms, and secondary rooms.
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :return: Bathroom constraint loss
        :rtype: torch.Tensor
        """
        neighbors_types = [RoomType.Entrance, RoomType.LivingRoom, RoomType.MasterRoom, RoomType.SecondRoom]

        return self._calculate_loss(plan_ids, RoomType.Bathroom, neighbors_types)


    def _balconies_loss(self, plan_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the balcony ergonomic constraint loss.
        
        Enforces that balconies are adjacent to living spaces (living rooms, study rooms, kitchens,
        and bedrooms).
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :return: Balcony constraint loss
        :rtype: torch.Tensor
        """
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

            for j, neighbor_id in enumerate(neighbors_idx):
                neighbor_coords = self._room_coords(plan_ids, neighbor_id)

                distances = torch.cdist(balcony_coords, neighbor_coords).view(interp_values_cnt, -1)
                loss = torch.sum(distances * F.softmin(distances * self.beta, -1), -1)

                losses[i, j, :] = loss

        losses = torch.sum(losses * F.softmin(losses * self.beta, 0), 0)
        losses = torch.sum(losses * F.softmin(losses * self.beta, 0), 0)

        return losses



    def _calculate_loss(self, plan_ids: torch.Tensor, main_type: RoomType, neighbors_types: list[RoomType]) -> torch.Tensor:
        """
        Calculates neighborhood loss between a main room type and neighbor room types.
        
        Computes pairwise distances between all instances of the main room and all neighbor rooms,
        then applies softmin weighting to create smooth differentiable constraints.
        
        :param plan_ids: Floor plan token IDs
        :type plan_ids: torch.Tensor
        :param main_type: The main room type to constrain
        :type main_type: RoomType
        :param neighbors_types: List of neighbor room types to constrain
        :type neighbors_types: list[RoomType]
        :return: Neighborhood constraint loss
        :rtype: torch.Tensor
        """
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

            for j, neighbor_id in enumerate(neighbors_idx):
                neighbor_coords = self._room_coords(plan_ids, neighbor_id)

                distances = torch.cdist(main_room_coords, neighbor_coords).view(interp_values_cnt, -1)
                loss = torch.sum(distances * F.softmin(distances * self.beta, -1), -1)

                losses[i, j, :] = loss

        losses = torch.sum(losses * F.softmin(losses * self.beta, 0), 0)
        losses = losses.mean(0)

        return losses
