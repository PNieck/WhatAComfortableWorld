import torch
import torch.nn.functional as F

import src.tokens as tokens

from src.floor_plan import RoomType


class KitchenEnforcementLoss:
    def __init__(self, base_loss, alpha = 0.3):
        self.base_loss = base_loss
        self.alpha = alpha
        self.kitchen_token_id = tokens.room_token_id(RoomType.Kitchen.value)


    def __call__(self, output, labels: torch.Tensor):
        std_loss = self.base_loss(output, labels)

        kitchen_token_mask = (labels == self.kitchen_token_id)
        seg_with_tokens = kitchen_token_mask.any(dim=1)
        if seg_with_tokens.any():
            return std_loss
        
        logits = output.logits

        probs = F.softmax(logits, dim=-1)[..., self.kitchen_token_id]

        log_probs = torch.log(probs + 1e-12)

        room_tokens_mask = self._is_room_token_id(labels)

        mask = room_tokens_mask
        mask[seg_with_tokens] = False
        mask = mask.float()

        encouragement_loss = -(log_probs * mask).sum() / (mask.sum() + 1e-12)

        total_loss = (1.0 - self.alpha) * std_loss + self.alpha * encouragement_loss

        return total_loss


    def _is_room_token_id(self, labels):
        return (labels >= tokens.CONT_TOKENS_CNT) & (labels <= tokens.CONT_TOKENS_CNT + tokens.ROOMS_CNT - 1)
