from .cross_entropy import CrossEntropyLoss
from .ignore_prompt import IgnorePromptInLoss
from .enforce_kitchen import KitchenEnforcementLoss
from .neighborhood_loss import NeighborhoodLoss
from .narrow_spaces import NarrowSpacesLoss


def _preprocess(training_config):
    if "ignore_prompt_in_loss" not in training_config:
        training_config["ignore_prompt_in_loss"] = False


def get_loss(training_config, tokenizer):
    if "loss" not in training_config:
        print("Warning - no specified loss - using default Cross Entropy")
        return CrossEntropyLoss()

    _preprocess(training_config)

    if training_config["loss"] == "KitchenEnforcementLoss":
        ce = CrossEntropyLoss()
        base_loss = KitchenEnforcementLoss(ce)

    elif training_config["loss"] == "CrossEntropyLoss":
        base_loss = CrossEntropyLoss()

    elif training_config["loss"] == "NeighborhoodLoss":
        ce = CrossEntropyLoss()
        base_loss = NeighborhoodLoss(ce)

    elif training_config["loss"] == "NarrowSpacesLoss":
        ce = CrossEntropyLoss()
        base_loss = NarrowSpacesLoss(ce, tokenizer)

    if training_config["ignore_prompt_in_loss"]:
        return IgnorePromptInLoss(base_loss)
    
    return base_loss
