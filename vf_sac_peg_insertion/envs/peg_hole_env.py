"""Peg-in-Hole Environment - DEBUGGED VERSION"""

import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import Env, spaces
from scipy.spatial.transform import Rotation
import os
import random
from collections import namedtuple

class PegHoleEnv(Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.physics_client = p.connect(p.GUI if config.render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(config.sim_timestep)
        
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1, -0.1]),
            high=np.array([0.1, 0.1, -0.02]),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'rgb_camera': spaces.Box(0, 255, (config.camera_height, config.camera_width, 3), dtype=np.uint8),
            'depth_camera': spaces.Box(0, 10, (config.camera_height, config.camera_width), dtype=np.float32),
            'force_torque_history': spaces.Box(-np.inf, np.inf, (30,), dtype=np.float32),
            'proprio': spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32),
            'goal': spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
        })
        
        self.robot_id = None
        self.peg_link_index = None
        self.camera_link_index = None
        self.hole_id = None
        self.table_id = None
        self.plane_id = None
        self.phase = 1
        self.current_step = 0
        self.force_history = np.zeros(30)
        self.joint_indices = []
        self.hole_clearance = 0.002
        
        JointInfo = namedtuple('jointInfo', ['id', 'name', 'type', 'lowerLimit', 'upperLimit',
                                              'maxForce', 'maxVelocity', 'controllable'])
        self.JointInfo = JointInfo
        self.joints = []
        self.controllable_joints = []
        
        self.image_width = config.camera_width
        self.image_height = config.camera_height
        
        self.last_joint_poses = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.config.sim_timestep)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            [0.4, 0, 0],
            p.getQuaternionFromEuler([0, 0, np.pi / 2])
        )
        
        if not os.path.exists(self.config.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.config.urdf_path}")
        
        self.robot_id = p.loadURDF(
            self.config.urdf_path,
            basePosition=[0, 0, 0.62],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        self._parse_joint_info()
        
        self.rest_poses = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        self.last_joint_poses = list(self.rest_poses)
        
        for i, joint_id in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=self.rest_poses[i],
                force=1000
            )
        
        for _ in range(100):
            p.stepSimulation()
        
        if hasattr(self, 'hole_id') and self.hole_id is not None:
            p.removeBody(self.hole_id)
        
        x = random.uniform(0.5, 0.6)
        y = random.uniform(0, 0.1)
        z = 0.65
        self.target_pos = np.array([x, y, z])
        
        if not os.path.exists(self.config.hole_mesh_path):
            raise FileNotFoundError(f"Hole mesh not found: {self.config.hole_mesh_path}")
        
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=self.config.hole_mesh_path,
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[0.6, 0.3, 0.1, 1.0]
        )
        
        hole_radius = self.config.peg_radius + self.hole_clearance
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=hole_radius,
            height=self.config.hole_depth
        )
        
        self.hole_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            baseCollisionShapeIndex=collision_shape,
            basePosition=self.target_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        self.force_history = np.zeros(30)
        self.phase = 1
        self.current_step = 0
        
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=110,
            cameraPitch=-45,
            cameraTargetPosition=[0.5, 0, 0.5]
        )
        
        return self._get_obs(), {}
    
    def _parse_joint_info(self):
        """Extract joint information"""
        self.joints = []
        self.controllable_joints = []
        self.joint_indices = []
        
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            lowerLimit = info[8]
            upperLimit = info[9]
            maxForce = info[10]
            maxVelocity = info[11]
            link_name = info[12].decode('utf-8')
            
            controllable = jointType != p.JOINT_FIXED
            
            self.joints.append(self.JointInfo(jointID, jointName, jointType, lowerLimit,
                                               upperLimit, maxForce, maxVelocity, controllable))
            
            if controllable:
                self.controllable_joints.append(jointID)
            
            if link_name == self.config.peg_link_name:
                self.peg_link_index = i
                print(f"[INFO] Found peg_link_index: {i}")
            
            if link_name == self.config.camera_link_name:
                self.camera_link_index = i
                print(f"[INFO] Found camera_link_index: {i}")
        
        self.joint_indices = self.controllable_joints[:6]
        
        print(f"[INFO] Joint indices: {self.joint_indices}")
        print(f"[INFO] Peg link index: {self.peg_link_index}")
        print(f"[INFO] Camera link index: {self.camera_link_index}")
    
    def step(self, action):
        """Execute action - COORDINATED JOINT MOVEMENT"""
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        peg_state = p.getLinkState(self.robot_id, self.peg_link_index)
        current_pos = np.array(peg_state[0])
        current_orn = peg_state[1]
        
        new_pos = current_pos + action
        new_pos = np.clip(new_pos, 
                        [0.2, -0.3, 0.4], 
                        [0.8, 0.3, 0.9])
        
        self._move_arm_to_fast(new_pos, current_orn)
        
        for _ in range(5):
            p.stepSimulation()
        
        obs = self._get_obs()
        
        reward = self._compute_reward(obs, action)
        
        if self.current_step % 20 == 0:
            print(f"[Step {self.current_step}] Reward: {reward:.3f}, Peg Position: x={current_pos[0]:.3f}, y={current_pos[1]:.3f}, z={current_pos[2]:.3f}")
        
        if self.phase == 1 and self._contact_detected():
            self.phase = 2
        
        obs['phase'] = np.array([self.phase], dtype=np.float32)
        
        done = self._check_done()
        truncated = self.current_step >= self.config.max_episode_steps
        
        info = {
            'phase': self.phase,
            'success': self._is_success(),
            'max_force': np.max(np.abs(obs['force_torque_history'][-6:-3])) if np.any(obs['force_torque_history']) else 0,
            'insertion_depth': self._get_insertion_depth(),
        }
        
        if done or truncated:
            print(f"[Episode End] Total Steps: {self.current_step}, Final Reward: {reward:.3f}, Final Peg Position: x={current_pos[0]:.3f}, y={current_pos[1]:.3f}, z={current_pos[2]:.3f}")
        
        return obs, reward, done, truncated, info


    def _move_arm_to_fast(self, pos, orn):
        """Move arm FAST using IK - all joints move together"""
        try:
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.peg_link_index,
                pos,
                orn,
                restPoses=list(self.last_joint_poses),
                lowerLimits=[self.joints[j].lowerLimit for j in self.joint_indices],
                upperLimits=[self.joints[j].upperLimit for j in self.joint_indices],
                jointRanges=[self.joints[j].upperLimit - self.joints[j].lowerLimit for j in self.joint_indices],
                maxNumIterations=50,
                residualThreshold=5e-3
            )
            
            self.last_joint_poses = list(joint_poses[:6])
            for i, joint_id in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=500, 
                    positionGain=0.01,
                    velocityGain=0.3
                )
            
        except Exception as e:
            print(f"[DEBUG] IK failed at {pos}, maintaining last pose")
            for i, joint_id in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=self.last_joint_poses[i],
                    force=500,
                    positionGain=0.1,
                    velocityGain=1.0
                )


    
    def _get_obs(self):
        """Get observation"""
        rgb, depth = self._render_eye_in_hand_camera()
        
        force_torque = self._read_virtual_ft_sensor()
        self.force_history = np.roll(self.force_history, -6)
        self.force_history[-6:] = force_torque
        
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        
        peg_state = p.getLinkState(self.robot_id, self.peg_link_index, computeLinkVelocity=1)
        peg_pos = np.array(peg_state[0])
        peg_orn = np.array(peg_state[1])
        
        proprio = np.concatenate([joint_positions, joint_velocities, peg_pos, peg_orn[:3]])
        
        hole_pos, hole_orn = p.getBasePositionAndOrientation(self.hole_id)
        goal = np.concatenate([hole_pos, hole_orn])
        
        return {
            'rgb_camera': rgb,
            'depth_camera': depth,
            'force_torque_history': self.force_history.copy(),
            'proprio': proprio,
            'goal': goal,
        }
    
    def _render_eye_in_hand_camera(self):
        """Render camera from link"""
        try:
            link_state = p.getLinkState(self.robot_id, self.camera_link_index, computeForwardKinematics=True)
            link_pos = link_state[0]
            link_ori = link_state[1]
            
            rot_matrix = p.getMatrixFromQuaternion(link_ori)
            rot_array = np.array(rot_matrix).reshape(3, 3)
            forward = -rot_array[:, 2]
            up = rot_array[:, 0]
            
            cam_eye = link_pos
            cam_target = [cam_eye[0] + forward[0] * 0.2,
                         cam_eye[1] + forward[1] * 0.2,
                         cam_eye[2] + forward[2] * 0.2]
            
            view = p.computeViewMatrix(cam_eye, cam_target, up)
            proj = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=self.image_width / self.image_height,
                nearVal=0.01,
                farVal=3.0
            )
            
            w, h, rgba, _, _ = p.getCameraImage(
                self.image_width,
                self.image_height,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=p.ER_TINY_RENDERER
            )
            
            rgba_img = np.reshape(rgba, (h, w, 4))
            rgb = rgba_img[:, :, :3]
            depth = rgba_img[:, :, 3] / 255.0 * 10.0
            
            return rgb.astype(np.uint8), depth.astype(np.float32)
        except Exception as e:
            print(f"[WARNING] Camera failed: {e}")
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8), \
                   np.zeros((self.image_height, self.image_width), dtype=np.float32)
    
    def _read_virtual_ft_sensor(self):
        """Read force-torque"""
        contact_points = p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.peg_link_index)
        
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        if len(contact_points) == 0:
            return np.concatenate([total_force, total_torque])
        
        for contact in contact_points:
            try:
                contact_normal = np.array(contact[7])
                normal_force = contact[9]
                contact_pos = np.array(contact[5])
                
                friction_force = 0.3 * normal_force * np.array([0, 0, -1])
                force = normal_force * contact_normal + friction_force
                total_force += force
                
                peg_state = p.getLinkState(self.robot_id, self.peg_link_index)
                peg_pos = np.array(peg_state[0])
                r = contact_pos - peg_pos
                torque = np.cross(r, force)
                total_torque += torque
            except:
                pass
        
        return np.concatenate([total_force, total_torque])
    
    def _contact_detected(self):
        """Check contact"""
        contacts = p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.peg_link_index)
        return len(contacts) > 0
    
    def _compute_reward(self, obs, action):
        """Compute reward with incremental guidance"""

        peg_pos = p.getLinkState(self.robot_id, self.peg_link_index)[0]
        hole_xy = np.array(self.target_pos[:2])
        peg_xy = np.array(peg_pos[:2])
        dist_xy = np.linalg.norm(peg_xy - hole_xy)

        dist_z = self.target_pos[2] - peg_pos[2]

        if not hasattr(self, 'prev_peg_z'):
            self.prev_peg_z = peg_pos[2]

        xy_radius = 0.1
        r_xy = np.clip((xy_radius - dist_xy) / xy_radius, 0, 1) * 5.0

        z_progress = max(0.0, self.prev_peg_z - peg_pos[2])
        z_bonus = 5.0 * z_progress

        if dist_xy < 0.04:
            z_bonus += 10.0 * np.clip(dist_z / 0.04, 0, 1)

        success_bonus = 100.0 if self._is_success() else 0.0

        action_penalty = -0.05 * np.linalg.norm(action)

        no_vert_progress_penalty = 0.0
        if dist_xy < 0.03 and abs(z_progress) < 1e-3:
            no_vert_progress_penalty = -0.5

        max_force = np.max(np.abs(obs['force_torque_history'][-6:-3]))
        force_penalty = -0.1 * max_force

        # Save peg z for next step comparison
        self.prev_peg_z = peg_pos[2]

        reward = r_xy + z_bonus + success_bonus + action_penalty + no_vert_progress_penalty + force_penalty
        return reward


        
    def _is_success(self):
        peg_pos = p.getLinkState(self.robot_id, self.peg_link_index)[0]
        dist_xy = np.linalg.norm(np.array(peg_pos[:2]) - np.array(self.target_pos[:2]))
        dist_z = self.target_pos[2] - peg_pos[2]
        return dist_xy < 0.02 and dist_z > 0.02


    
    def _get_insertion_depth(self):
        """Get insertion depth"""
        peg_pos = p.getLinkState(self.robot_id, self.peg_link_index)[0]
        insertion_depth = max(0, self.target_pos[2] - peg_pos[2])
        return min(insertion_depth, self.config.peg_length)
    
    def _check_done(self):
        """Check if done"""
        return self._is_success()
    
    def set_curriculum_stage(self, stage_config):
        """Update curriculum"""
        self.config.curriculum_stage = {'simple': 1, 'moderate': 2, 'complex': 3}.get(stage_config['name'], 1)
        self.hole_clearance = stage_config.get('hole_clearance', 0.002)
    
    def render(self):
        pass
    
    def close(self):
        p.disconnect(self.physics_client)