from .cross_entropy import CrossEntropyLoss
from .ignore_prompt import IgnorePromptInLoss
from .enforce_kitchen import KitchenEnforcementLoss
from .neighborhood_loss import NeighborhoodLoss
from .narrow_spaces import NarrowSpacesLoss
from .mean_values_loss import MeanValuesLoss
from .segments_dist_loss import SegmentDistLoss
from .std_loss import StdLoss

from src.training_config import TrainingConfig


def get_loss(config: TrainingConfig, tokenizer, train_dataloader, device):
    match config.training_loss:
        case "KitchenEnforcementLoss":
            ce = CrossEntropyLoss()
            base_loss = KitchenEnforcementLoss(ce)

        case "CrossEntropyLoss":
            base_loss = CrossEntropyLoss()

        case "NeighborhoodLoss":
            base_loss = NeighborhoodLoss(device)
            # base_loss.update_max_ergo_loss(train_dataloader, device)

        case "MeanValuesLoss":
            base_loss = MeanValuesLoss(device)

        case "SegmentDistLoss":
            base_loss = SegmentDistLoss(device)

        case "StdLoss":
            base_loss = StdLoss()

        case "NarrowSpacesLoss":
            ce = CrossEntropyLoss()
            base_loss = NarrowSpacesLoss(ce, tokenizer)

        case _:
            raise ValueError(f"Unexpected loss type {config.training_loss}")
        
    if config.ignore_prompt_in_loss:
        return IgnorePromptInLoss(base_loss)
    
    return base_loss
