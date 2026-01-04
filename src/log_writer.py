from torch.utils.tensorboard import SummaryWriter

import os
import json





class LogWriter:
    def __init__(self, log_dir: str):       
        self.start_epoch = 0
        self.start_step = 0

        self.sw = SummaryWriter(log_dir)

    @classmethod
    def from_checkpoint(cls, log_dir, checkpoint_epoch):
        result = LogWriter(log_dir)

        checkpoint_dir = os.path.join(log_dir, "checkpoints", f"epoch_{checkpoint_epoch}")
        if not os.path.isdir(checkpoint_dir):
            raise Exception(f"No checkpoint dir {checkpoint_dir}")
        
        training_status_path =  os.path.join(checkpoint_dir, "training_status.json")
        if not os.path.exists(training_status_path):
            raise Exception(f"No training status file {training_status_path}")
        
        with open(training_status_path, "r") as file:
            training_status = json.load(file)
            result.start_epoch = training_status.get("epochs")
            result.start_step = training_status.get("steps")

        return result


    def add_scalar(self, tag: str, scalar_value: float, step: int):
        self.sw.add_scalar(tag, scalar_value, step + self.start_step)


    def add_hparams(self, hparam_dict: dict, metric_dict: dict):
        self.sw.add_hparams(hparam_dict, metric_dict)


    def state_dict(self, epoch, step) -> dict:
        return {
            "epochs": epoch,
            "steps": step + self.start_step
        }
    
    def load_state_dict(self, dict: dict):
        self.start_epoch = dict["epochs"]
        self.start_step = dict["steps"]

