"""SAC Algorithm Configuration"""

class SACConfig:
    # Network architecture
    visual_feature_dim = 128
    force_feature_dim = 128
    fused_feature_dim = 256
    proprio_dim = 18  # 6 joint angles + 6 velocities + 6 EE pose components
    action_dim = 6
    
    # SAC hyperparameters
    actor_lr = 3e-4
    critic_lr = 3e-4
    encoder_lr = 1e-4
    alpha_lr = 3e-4
    gamma = 0.99
    tau = 0.005
    alpha_init = 0.2
    auto_tune_alpha = True
    target_entropy = -action_dim
    
    # Training parameters
    batch_size = 256
    buffer_size = int(1e6)
    warmup_steps = 5000
    updates_per_step = 1
    
    # Replay buffer
    her_strategy = 'future'
    her_ratio = 0.8
    
    # Force thresholds
    max_force_threshold = 7.5  # Newtons
    max_torque_threshold = 0.5  # N·m
