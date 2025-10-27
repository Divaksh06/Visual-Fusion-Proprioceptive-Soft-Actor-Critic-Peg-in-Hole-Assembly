"""Test/Evaluation Script"""

import torch
import numpy as np
import argparse

from config.sac_config import SACConfig
from config.env_config import EnvConfig
from envs.peg_hole_env import PegHoleEnv
from models.encoders import VisionEncoder, ForceEncoder
from models.attention import CrossModalAttention
from models.policy import ActorNetwork
from models.value import CriticNetwork
from algorithms.sac import SAC

def main(args):
    # Configuration
    sac_config = SACConfig()
    env_config = EnvConfig()
    env_config.render = True  # Always render during testing
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Testing on device: {device}")
    
    # Environment
    env = PegHoleEnv(env_config)
    
    # Networks
    vision_encoder = VisionEncoder(feature_dim=sac_config.visual_feature_dim)
    force_encoder = ForceEncoder(feature_dim=sac_config.force_feature_dim)
    cross_modal_attention = CrossModalAttention(feature_dim=sac_config.visual_feature_dim, num_heads=4)
    
    state_dim = sac_config.fused_feature_dim + sac_config.proprio_dim + 1
    actor = ActorNetwork(state_dim, sac_config.action_dim)
    critic1 = CriticNetwork(state_dim, sac_config.action_dim)
    critic2 = CriticNetwork(state_dim, sac_config.action_dim)
    
    # SAC Agent
    agent = SAC(
        config=sac_config,
        vision_encoder=vision_encoder,
        force_encoder=force_encoder,
        attention_module=cross_modal_attention,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        device=device
    )
    
    # Load checkpoint
    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        agent.vision_encoder.load_state_dict(checkpoint['vision_encoder'])
        agent.force_encoder.load_state_dict(checkpoint['force_encoder'])
        agent.attention_module.load_state_dict(checkpoint['attention_module'])
        agent.actor.load_state_dict(checkpoint['actor'])
        print(f"[INFO] Loaded checkpoint from episode {checkpoint['episode']}")
    
    # Test
    success_count = 0
    
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        obs['phase'] = np.array([1.0], dtype=np.float32)
        episode_reward = 0
        
        print(f"\n[INFO] Testing Episode {episode+1}/{args.num_episodes}")
        
        for step in range(500):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        success = info.get('success', False)
        success_count += int(success)
        
        print(f"  Result: {'SUCCESS' if success else 'FAILURE'}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {step+1}")
        print(f"  Insertion Depth: {info.get('insertion_depth', 0):.4f}m")
        print(f"  Max Force: {info.get('max_force', 0):.2f}N")
    
    print(f"\n[SUMMARY] Success Rate: {success_count}/{args.num_episodes} = {success_count/args.num_episodes:.2%}")
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of test episodes')
    
    args = parser.parse_args()
    main(args)
