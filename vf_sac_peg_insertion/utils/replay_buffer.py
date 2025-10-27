"""Simple Replay Buffer (backup to HER)"""

import numpy as np
import torch

class ReplayBuffer:
    """Simple experience replay buffer"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self):
        return len(self.buffer)
