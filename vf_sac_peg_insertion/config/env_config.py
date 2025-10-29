"""Environment Configuration for UR5 Robotiq System"""

class EnvConfig:
    # Robot configuration
    urdf_path = "DRL_Peg-in-Hole_UR5/urdf/ur5_robotiq_85.urdf"
    hole_mesh_path = "DRL_Peg-in-Hole_UR5/urdf/box.stl"
    
    # UR5 joint configuration
    ur5_joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint", 
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]
    num_joints = 6
    
    # End-effector and peg links
    ee_link_name = "ee_link"
    peg_link_name = "cylinder_link"
    camera_link_name = "camera_link"
    
    # Camera configuration (single eye-in-hand)
    camera_width = 256
    camera_height = 256
    camera_fov = 60
    camera_near = 0.01
    camera_far = 2.0
    
    # Force-torque sensor simulation
    ft_sensor_joint = "wrist_3_joint"
    ft_history_length = 5
    
    # Peg properties (from URDF - cylindrical)
    peg_radius = 0.018  # meters (18mm)
    peg_length = 0.08   # meters (80mm)
    
    # Hole properties (cylindrical from box.stl)
    hole_radius = 0.020  # meters (20mm) - 2mm clearance
    hole_depth = 0.085   # meters (slightly deeper than peg)
    
    # Simulation
    sim_timestep = 1/240.0
    sim_steps_per_action = 10
    max_episode_steps = 60
    
    # Control
    velocity_control = True
    max_joint_velocity = 1.0  # rad/s
    max_cartesian_velocity = 0.25  # m/s
    max_angular_velocity = 1.57  # rad/s (90 deg/s)
    
    # Curriculum
    curriculum_stage = 1
    render = False
    
    # Workspace limits
    workspace_center = [0.3, 0.0, 0.3]
    workspace_radius = 0.5


