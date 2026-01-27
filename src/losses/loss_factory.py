from .cross_entropy import CrossEntropyLoss
from .ignore_prompt import IgnorePromptInLoss
from .neighborhood_loss import NeighborhoodLoss
from .mean_values_loss import MeanValuesLoss
from .segments_dist_loss import SegmentDistLoss
from .std_loss import StdLoss

from src.training_config import TrainingConfig


def get_loss(config: TrainingConfig):
    match config.training_loss:
        case "CrossEntropyLoss":
            base_loss = CrossEntropyLoss()

        case "NeighborhoodLoss":
            base_loss = NeighborhoodLoss()

        case "MeanValuesLoss":
            base_loss = MeanValuesLoss()

        case "SegmentDistLoss":
            base_loss = SegmentDistLoss()

        case "StdLoss":
            base_loss = StdLoss()

        case _:
            raise ValueError(f"Unexpected loss type {config.training_loss}")
        
    if config.ignore_prompt_in_loss:
        return IgnorePromptInLoss(base_loss)
    
    return base_loss
