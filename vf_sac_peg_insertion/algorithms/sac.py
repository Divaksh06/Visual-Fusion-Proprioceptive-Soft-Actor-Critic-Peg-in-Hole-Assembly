"""Soft Actor-Critic (SAC) Algorithm"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

class SAC:
    def __init__(self, config, vision_encoder, force_encoder, attention_module,
                 actor, critic1, critic2, device='cuda'):
        self.config = config
        self.device = device
        
        self.vision_encoder = vision_encoder.to(device)
        self.force_encoder = force_encoder.to(device)
        self.attention_module = attention_module.to(device)
        self.actor = actor.to(device)
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        
        self.target_critic1 = deepcopy(critic1).to(device)
        self.target_critic2 = deepcopy(critic2).to(device)
        
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()),
            lr=config.critic_lr
        )
        self.encoder_optimizer = optim.Adam(
            list(vision_encoder.parameters()) +
            list(force_encoder.parameters()) +
            list(attention_module.parameters()),
            lr=config.encoder_lr
        )
        
        if config.auto_tune_alpha:
            self.log_alpha = torch.tensor(
                np.log(config.alpha_init), requires_grad=True, device=device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.target_entropy = config.target_entropy
        else:
            self.log_alpha = torch.tensor(np.log(config.alpha_init), device=device)
        
        self.training_step = 0
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def encode_state(self, obs):
        """Encode multimodal observation"""
        rgb = obs['rgb_camera'].to(self.device).float() / 255.0
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)
        
        depth = obs['depth_camera'].to(self.device)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)
        
        φ_v = self.vision_encoder(rgb, depth)
        
        force_history = obs['force_torque_history'].to(self.device)
        if force_history.dim() == 1:
            force_history = force_history.unsqueeze(0)
        φ_f = self.force_encoder(force_history)
        
        φ_fused, attention_weights = self.attention_module(φ_v, φ_f)
        
        proprio = obs['proprio'].to(self.device)
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        
        phase = obs['phase'].to(self.device)
        if phase.dim() == 1:
            phase = phase.unsqueeze(0)
        
        state = torch.cat([φ_fused, proprio, phase], dim=-1)
        
        return state, attention_weights
    
    def select_action(self, obs, deterministic=False):
        """Select action from policy"""
        with torch.no_grad():
            state, _ = self.encode_state(obs)
            if deterministic:
                action = self.actor.deterministic_action(state)
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]
    
    def update(self, batch):
        """Perform one SAC update step"""
        state, _ = self.encode_state(batch['obs'])
        next_state, _ = self.encode_state(batch['next_obs'])
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        done = batch['done'].to(self.device)
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = reward + self.config.gamma * (1 - done) * (
                target_q - self.alpha * next_log_prob
            )
        
        q1_pred = self.critic1(state, action)
        q2_pred = self.critic2(state, action)
        critic_loss = F.mse_loss(q1_pred, target_value) + F.mse_loss(q2_pred, target_value)
        
        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()
        
        for param in self.critic1.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
        
        new_action, new_log_prob = self.actor.sample(state.detach())
        q1_new = self.critic1(state.detach(), new_action)
        q2_new = self.critic2(state.detach(), new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha.detach() * new_log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param in self.critic1.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True
        
        if self.config.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)
        
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q_value': q1_pred.mean().item(),
        }
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )
