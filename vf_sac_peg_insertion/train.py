"""Main Training Script"""

import torch
import numpy as np
from tqdm import tqdm
import argparse
import os

from config.sac_config import SACConfig
from config.env_config import EnvConfig
from config.curriculum_config import CurriculumConfig
from envs.peg_hole_env import PegHoleEnv
from models.encoders import VisionEncoder, ForceEncoder
from models.attention import CrossModalAttention
from models.policy import ActorNetwork
from models.value import CriticNetwork
from algorithms.sac import SAC
from algorithms.her import HERBuffer
from algorithms.curriculum import CurriculumManager
from utils.logger import Logger

def main(args):
    sac_config = SACConfig()
    env_config = EnvConfig()
    env_config.render = args.render
    curriculum_config = CurriculumConfig()
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"[INFO] Training on device: {device}")
    
    env = PegHoleEnv(env_config)
    
    vision_encoder = VisionEncoder(feature_dim=sac_config.visual_feature_dim)
    force_encoder = ForceEncoder(feature_dim=sac_config.force_feature_dim)
    cross_modal_attention = CrossModalAttention(
        feature_dim=sac_config.visual_feature_dim,
        num_heads=4
    )
    
    state_dim = sac_config.fused_feature_dim + sac_config.proprio_dim + 1
    actor = ActorNetwork(state_dim, sac_config.action_dim)
    critic1 = CriticNetwork(state_dim, sac_config.action_dim)
    critic2 = CriticNetwork(state_dim, sac_config.action_dim)
    
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
    
    replay_buffer = HERBuffer(
        capacity=int(sac_config.buffer_size),
        her_ratio=sac_config.her_ratio,
        strategy=sac_config.her_strategy
    )
    
    curriculum = CurriculumManager(curriculum_config)
    
    logger = Logger(args.log_dir)
    
    total_steps = 0
    episode = 0
    
    print(f"[INFO] Starting training for {args.total_episodes} episodes")
    pbar = tqdm(total=args.total_episodes, desc="Training")
    
    while episode < args.total_episodes and not curriculum.is_complete():
        stage_config = curriculum.get_current_stage()
        env.set_curriculum_stage(stage_config)
        
        obs, info = env.reset()
        obs['phase'] = np.array([1.0], dtype=np.float32)
        episode_transitions = []
        episode_reward = 0
        episode_length = 0
        
        for t in range(args.max_episode_length):
            if total_steps < sac_config.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=False)
            
            next_obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            episode_transitions.append((obs, action, reward, next_obs, done, info))
            obs = next_obs
            
            if done or truncated:
                break
        
        replay_buffer.add_episode(episode_transitions)
        
        curriculum.update(info.get('success', False))
        
        if total_steps >= sac_config.warmup_steps and len(replay_buffer) > sac_config.batch_size:
            for _ in range(episode_length):
                batch = replay_buffer.sample(sac_config.batch_size)
                train_metrics = agent.update(batch)
                if _ == 0:
                    logger.log_train_metrics(train_metrics, total_steps)
        
        logger.log_episode_metrics({
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success': info.get('success', False),
            'max_force': info.get('max_force', 0.0),
            'insertion_depth': info.get('insertion_depth', 0.0),
            'curriculum_stage': curriculum.current_stage,
        }, episode)
        
        episode += 1
        pbar.update(1)
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'success': info.get('success', False),
            'stage': curriculum.current_stage
        })
        
        if episode % args.eval_interval == 0:
            eval_metrics = evaluate(agent, env, n_episodes=10)
            logger.log_eval_metrics(eval_metrics, episode)
            print(f"\n[EVAL] Episode {episode}: Success Rate = {eval_metrics['success_rate']:.2%}")
        
        if episode % args.save_interval == 0:
            save_checkpoint(agent, episode, args.save_dir)
            print(f"[INFO] Checkpoint saved at episode {episode}")
    
    pbar.close()
    logger.close()
    env.close()
    print("\n[INFO] Training completed!")

def evaluate(agent, env, n_episodes=10):
    """Evaluate agent"""
    success_count = 0
    total_rewards = []
    max_forces = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs['phase'] = np.array([1.0], dtype=np.float32)
        episode_reward = 0
        
        for _ in range(500):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        success_count += int(info.get('success', False))
        total_rewards.append(episode_reward)
        max_forces.append(info.get('max_force', 0.0))
    
    return {
        'success_rate': success_count / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'mean_max_force': np.mean(max_forces),
    }

def save_checkpoint(agent, episode, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'episode': episode,
        'vision_encoder': agent.vision_encoder.state_dict(),
        'force_encoder': agent.force_encoder.state_dict(),
        'attention_module': agent.attention_module.state_dict(),
        'actor': agent.actor.state_dict(),
        'critic1': agent.critic1.state_dict(),
        'critic2': agent.critic2.state_dict(),
        'log_alpha': agent.log_alpha,
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint_ep{episode}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_episodes', type=int, default=100000)
    parser.add_argument('--max_episode_length', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--render', action='store_true', default=False)
    
    args = parser.parse_args()
    main(args)
