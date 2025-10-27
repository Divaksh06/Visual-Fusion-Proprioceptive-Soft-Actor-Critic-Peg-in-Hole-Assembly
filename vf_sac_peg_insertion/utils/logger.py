"""TensorBoard Logger"""

import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """Logger for training metrics"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
    def log_train_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value, step)
    
    def log_episode_metrics(self, metrics, episode):
        for key, value in metrics.items():
            self.writer.add_scalar(f'episode/{key}', value, episode)
    
    def log_eval_metrics(self, metrics, episode):
        for key, value in metrics.items():
            self.writer.add_scalar(f'eval/{key}', value, episode)
    
    def close(self):
        self.writer.close()
