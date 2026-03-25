import torch
import torch.nn.functional as F

import src.tokens as tokens
from src.floor_plan import RoomType
from .base_ergo_loss import BaseErgoLoss


class SegmentDistErgoLoss(BaseErgoLoss):
    """
    Computes a weighted combination of standard cross-entropy loss and ergonomic loss.
    
    This loss function enforces neighborhood relationships between rooms in floor plans by
    calculating distances between related room types (e.g., kitchens near entrances, bathrooms
    near living areas). Uses adaptive weighting based on ergonomic score.

    Neighborhood is calculated based on distance between line segments of 
    room's polygons.
    """
    def __init__(self):
        super().__init__(scaling=1_000)

        self.beta = 10.0

        self.max_ergo_loss = torch.ones(1, device="cpu") * 2_000.0


    def __call__(self, output, labels: torch.Tensor):
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

        door_starts = door_points[None, None, 0]
        door_ends = door_points[None, None, 1]

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
    

    def point_segment_distance_squared(self, p, a, b, eps=1e-9):
        """
        p : (..., D)
        a,b : (..., D)
        returns: (...,)
        """
        ab = b - a
        ap = p - a

        t = (ap * ab).sum(dim=-1) / ((ab * ab).sum(dim=-1) + eps)
        t = t.clamp(0.0, 1.0)

        closest = a + t[..., None] * ab
        diff = p - closest
        return (diff * diff).sum(dim=-1)
    

    def segment_segment_distance_double_batch(
        self, p, p2, q, q2, eps=1e-9
    ):
        u = p2 - p
        v = q2 - q

        # Expand for pairwise
        p_ = p[:, :, None, :]
        q_ = q[:, None, :, :]
        u_ = u[:, :, None, :]
        v_ = v[:, None, :, :]

        w = p_ - q_

        a = (u_ * u_).sum(dim=-1)
        b = (u_ * v_).sum(dim=-1)
        c = (v_ * v_).sum(dim=-1)
        d = (u_ * w).sum(dim=-1)
        e = (v_ * w).sum(dim=-1)

        D = a * c - b * b

        # ---------- non-parallel case ----------
        t = (b * e - c * d) / (D + eps)
        s = (a * e - b * d) / (D + eps)

        t = t.clamp(0.0, 1.0)
        s = s.clamp(0.0, 1.0)

        cp = p_ + t[..., None] * u_
        cq = q_ + s[..., None] * v_

        dist2_np = ((cp - cq) ** 2).sum(dim=-1)

        # ---------- parallel case fallback ----------
        d1 = self.point_segment_distance_squared(p_ , q_ , q_ + v_, eps)
        d2 = self.point_segment_distance_squared(p_ + u_, q_ , q_ + v_, eps)
        d3 = self.point_segment_distance_squared(q_ , p_ , p_ + u_, eps)
        d4 = self.point_segment_distance_squared(q_ + v_, p_ , p_ + u_, eps)

        dist2_parallel = torch.minimum(
            torch.minimum(d1, d2),
            torch.minimum(d3, d4),
        )

        # ---------- select based on parallelism ----------
        parallel = D.abs() < eps

        dist2 = torch.where(parallel, dist2_parallel, dist2_np)

        return dist2
