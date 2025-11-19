from torch.utils.tensorboard import SummaryWriter

import os
import json


class LogWriter:
    def __init__(self, log_dir: str):
        training_status_path = f"{log_dir}/training_status.json"

        if os.path.isdir(log_dir) and os.path.exists(training_status_path):
            with open(training_status_path, "r") as file:
                training_status = json.load(file)
                self.start_epoch = training_status.get("epochs")
                self.start_step = training_status.get("steps")
        else:
            self.start_epoch = 0
            self.start_step = 0

        self.sw = SummaryWriter(log_dir)


    def add_scalar(self, tag: str, scalar_value: float, step: int):
        self.sw.add_scalar(tag, scalar_value, step + self.start_step)


    def add_hparams(self, hparam_dict: dict, metric_dict: dict):
        self.sw.add_hparams(hparam_dict, metric_dict)


    def save_training_status(self, step: int, epoch: int, dir: str = None):
        if dir is None:
            dir = self.sw.log_dir
        
        path = dir + "/training_status.json"
        
        data = {
            "epochs": epoch + self.start_epoch,
            "steps": step + self.start_step
        }

        json_str = json.dumps(data, indent=4)

        with open(path, "w") as file:
            file.write(json_str)

