import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import os
import datetime
from dataclasses import dataclass


@dataclass
class CyclicLRSchedulerConfig:
    base_lr: float
    max_lr: float
    cycles: int


class TrainingConfig:

    @property
    def seed(self) -> int | None:
        if "seed" not in self.general_config:
            return None
        
        return int(self.general_config["seed"])
    
    @property
    def model_has_xy_indices(self) -> bool:
        return self.model_config["with_xy_indices"]
    
    @property
    def model_has_corner_indices(self) -> bool:
        return self.model_config["with_corner_indices"]
    
    @property
    def start_with_pretrained_model(self) -> bool:
        return "input_model_path" in self.model_config
    
    @property
    def start_pretrained_model_path(self) -> str:
        """
        Returns only if `start_with_pretrained_model` returns True
        """
        return self.model_config["input_model_path"]
    
    @property
    def vocab_size(self) -> int:
        return self.model_config["vocab_size"]
    
    @property
    def max_sequence_len(self) -> int:
        return int(self.model_config["max_seq_len"])
    
    @property
    def model_layers_cnt(self) -> int:
        return int(self.model_config["n_layer"])
    
    @property
    def model_heads_cnt(self) -> int:
        return int(self.model_config["n_head"])
    
    @property
    def model_embedding_dim(self) -> int:
        return int(self.model_config["n_embd"])
    
    @property
    def input_data_path(self) -> str:
        return self.train_config["input_data"]
    
    @property
    def ignore_prompt_in_loss(self) -> bool:
        return self.train_config["ignore_prompt_in_loss"]
    
    @property
    def training_loss(self) -> str:
        return self.train_config["loss"]
    
    @property
    def epochs_cnt(self) -> int:
        return int(self.train_config["epochs"])
    
    @property
    def lr_scheduler_type(self) -> str:
        return self.train_config["lr_scheduler"]["type"]
    
    @property
    def lr_scheduler_config(self) -> CyclicLRSchedulerConfig | None:
        return self.lr_config

    @property
    def eval_steps(self) -> int:
        return int(self.train_config["eval_steps"])
    
    @property
    def batch_size(self) -> int:
        return int(self.train_config["batch_size"])
    
    @property
    def learning_rate(self) -> float:
        return float(self.train_config["lr"])

    @property
    def checkpointing_frequency(self) -> int:
        return int(self.train_config["checkpointing_frequency"])
    
    @property
    def use_checkpoint(self) -> bool:
        return self.checkpoint_path is not None


    def __init__(self, path: str|None = None):
        if path is None:
            self.general_config = {}
            self.model_config = {}
            self.train_config = {}
            self.lr_config = None
            self.log_dir = ""  
            return

        with open(path, "r") as f:
            config = yaml.load(f, Loader=Loader)

        self.load_state_dict(config)

        self.checkpoint_path = None
        self.checkpoint_epoch = None


    def update_with_tokenizer(self, tokenizer):
        self.model_config["vocab_size"] = len(tokenizer)


    def state_dict(self) -> dict:
        result = {}
        result["general"] = self.general_config
        result["model"] = self.model_config
        result["training"] = self.train_config
        result["logging"] = self.log_dir

        return result
    
    def load_state_dict(self, d: dict):
        if "general" not in d:
            self.general_config: dict = dict()
        else:
            self.general_config: dict = d["general"]

        self.model_config: dict = d["model"]
        self.train_config: dict = d["training"]

        self.lr_config = None

        self._preprocess_model_config()
        self._preprocess_train_config()

        if "logging" in d:
            self.log_dir = d["logging"]
        else:
            self.log_dir: str = self._get_log_dir()

    
    def _preprocess_model_config(self):
        TrainingConfig._preprocess_bool_entry(self.model_config, "with_xy_indices")
        TrainingConfig._preprocess_bool_entry(self.model_config, "with_corner_indices")


    def _preprocess_train_config(self):
        TrainingConfig._preprocess_bool_entry(self.train_config, "ignore_prompt_in_loss")

        if "loss" not in self.train_config:
            print("Warning - no specified loss - using default Cross Entropy")
            self.train_config["loss"] = "CrossEntropyLoss"

        if "lr_scheduler" not in self.train_config:
            print("Learning scheduler not specified - using linear one")
            self.train_config["lr_scheduler"] = {}
            self.train_config["lr_scheduler"]["type"] = "linear"

        if self.train_config["lr_scheduler"]["type"] == "cyclic":
            scheduler_config = self.train_config["lr_scheduler"]
            
            self.lr_config = CyclicLRSchedulerConfig(
                float(scheduler_config["base_lr"]),
                float(scheduler_config["max_lr"]),
                int(scheduler_config["cycles"])
            )


    @staticmethod
    def _preprocess_bool_entry(config: dict, entry: str):
        if entry not in config:
            config[entry] = False


    def _get_log_dir(self) -> str:
        result = os.path.join("runs", datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))

        if "log_comment" in self.train_config:
            result += self.train_config["log_comment"]

        return result
