"""Hindsight Experience Replay"""

import numpy as np
import torch

class HERBuffer:
    """Experience replay buffer with HER"""
    def __init__(self, capacity, her_ratio=0.8, strategy='future'):
        self.capacity = capacity
        self.her_ratio = her_ratio
        self.strategy = strategy
        self.buffer = []
        self.position = 0
    
    def add_episode(self, episode_transitions):
        """Add episode with HER augmentation"""
        for transition in episode_transitions:
            self._add_transition(transition)
        
        if self.strategy == 'future':
            her_transitions = self._apply_future_strategy(episode_transitions)
            for transition in her_transitions:
                self._add_transition(transition)
    
    def _add_transition(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def _apply_future_strategy(self, episode_transitions):
        """HER 'future' strategy"""
        her_transitions = []
        
        for i in range(len(episode_transitions) - 1):
            obs, action, _, next_obs, _, info = episode_transitions[i]
            
            num_samples = min(4, len(episode_transitions) - i - 1)
            if num_samples == 0:
                continue
            
            future_indices = np.random.choice(
                range(i+1, len(episode_transitions)),
                size=num_samples,
                replace=False
            )
            
            for future_idx in future_indices:
                future_next_obs = episode_transitions[future_idx][3]
                new_goal = future_next_obs['proprio'][:7]
                
                new_reward = -np.linalg.norm(next_obs['proprio'][:3] - new_goal[:3])
                new_done = np.linalg.norm(next_obs['proprio'][:3] - new_goal[:3]) < 0.01
                
                new_obs = obs.copy()
                new_obs['goal'] = np.concatenate([new_goal, [0, 0, 0, 1]])
                new_next_obs = next_obs.copy()
                new_next_obs['goal'] = np.concatenate([new_goal, [0, 0, 0, 1]])
                
                her_transitions.append((
                    new_obs, action, new_reward, new_next_obs, new_done, info
                ))
        
        return her_transitions
    
    def sample(self, batch_size):
        """Sample batch of transitions"""
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[idx] for idx in indices]
        
        obs_batch = {}
        next_obs_batch = {}
        action_batch = []
        reward_batch = []
        done_batch = []
        
        for transition in batch:
            obs, action, reward, next_obs, done, _ = transition
            
            for key in obs.keys():
                if key not in obs_batch:
                    obs_batch[key] = []
                    next_obs_batch[key] = []
                obs_batch[key].append(obs[key])
                next_obs_batch[key].append(next_obs[key])
            
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)
        
        for key in obs_batch.keys():
            obs_batch[key] = torch.tensor(np.array(obs_batch[key]), dtype=torch.float32)
            next_obs_batch[key] = torch.tensor(np.array(next_obs_batch[key]), dtype=torch.float32)
        
        return {
            'obs': obs_batch,
            'next_obs': next_obs_batch,
            'action': torch.tensor(action_batch, dtype=torch.float32),
            'reward': torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(-1),
            'done': torch.tensor(done_batch, dtype=torch.float32).unsqueeze(-1),
        }
    
    def __len__(self):
        return len(self.buffer)
