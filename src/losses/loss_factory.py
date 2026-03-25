"""Factory for creating loss functions based on configuration.

Provides factory function for instantiating loss functions with optional
prompt masking wrapper based on training configuration.
"""

from .cross_entropy import CrossEntropyLoss
from .ignore_prompt import IgnorePromptInLoss
from .vertex_dist_ergo_loss import VertexDistancesErgoLoss
from .mean_vertex_dist_ergo_loss import MeanVertexDistErgoLoss
from .segments_dist_ergo_loss import SegmentDistErgoLoss

from src.training_config import TrainingConfig


def get_loss(config: TrainingConfig):
    """Create loss function based on training configuration.
    
    Instantiates the appropriate loss function based on config.training_loss
    and optionally wraps it with IgnorePromptInLoss if enabled.
    
    :param config: Training configuration specifying loss type
    :return: Loss function object
    :raises ValueError: If training_loss type is not recognized
    """

    match config.training_loss:
        case "CrossEntropyLoss":
            base_loss = CrossEntropyLoss()

        case "VertexDistancesErgoLoss":
            base_loss = VertexDistancesErgoLoss()

        case "MeanVertexDistErgoLoss":
            base_loss = MeanVertexDistErgoLoss()

        case "SegmentDistErgoLoss":
            base_loss = SegmentDistErgoLoss()

        case _:
            raise ValueError(f"Unexpected loss type {config.training_loss}")
        
    if config.ignore_prompt_in_loss:
        return IgnorePromptInLoss(base_loss)
    
    return base_loss
