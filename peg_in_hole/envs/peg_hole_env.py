"""
Main MuJoCo Gymnasium environment for the peg-in-hole task.

Key design points
-----------------
* Insertion axis is **+Y** (world), NOT -Z.
* peg_tip site position is the ground-truth peg tip — never use body("peg").xpos.
* F/T sensor readings are rotated from sensor frame to world frame before use.
* Gripper stays closed (ctrl[6] = 50.0) throughout the episode.
* PyRoKi handles IK; MuJoCo handles physics + sensors.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from peg_in_hole.kinematics.pyroki_ik import HealIK
from peg_in_hole.envs.reward import compute_reward
from peg_in_hole.envs.observation_builder import ObservationBuilder
from peg_in_hole.envs.curriculum import CurriculumManager


# ---------------------------------------------------------------------------
# MuJoCo name constants
# ---------------------------------------------------------------------------
PEG_BODY = "peg"
HOLE_BODY = "hole_assembly"
FT_SENSOR_BODY = "force_torque_sensor"

PEG_TIP_SITE = "peg_tip"
FT_SITE = "ft_sensor_site"

FT_FORCE_SENSOR = "ft_force_sensor"
FT_TORQUE_SENSOR = "ft_torque_sensor"

CAMERA_NAME = "realsense_rgb"

ARM_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

INSERTION_AXIS = np.array([0.0, 1.0, 0.0])   # +Y world frame
HOLE_WORLD_POS = np.array([0.5, 0.0, 0.425])  # approximate; refreshed from sim


class PegInHoleEnv(gym.Env):
    """Peg-in-hole insertion environment with multimodal observations."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, cfg: dict, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        # ---- MuJoCo ----
        self.mj_model = mujoco.MjModel.from_xml_path(cfg["xml_path"])
        self.mj_data = mujoco.MjData(self.mj_model)

        # Override timestep if not already set in XML
        self.mj_model.opt.timestep = cfg.get("sim_timestep", 0.002)
        self.decimation = cfg.get("control_decimation", 5)
        self.max_steps = cfg.get("episode_length", 500)

        # ---- PyRoKi IK ----
        self.heal_ik = HealIK(urdf_path=cfg["urdf_path"])

        # ---- PD gains ----
        self.kp = np.array(cfg.get("pd_kp", [80, 80, 80, 40, 40, 40]), dtype=np.float64)
        self.kd = np.array(cfg.get("pd_kd", [10, 10, 10, 5, 5, 5]), dtype=np.float64)
        self.gripper_ctrl = cfg.get("gripper_ctrl", 50.0)

        self.ctrl_lo = np.array([-100, -100, -100, -50, -50, -50], dtype=np.float64)
        self.ctrl_hi = np.array([ 100,  100,  100,  50,  50,  50], dtype=np.float64)

        # ---- Observation builder ----
        self.obs_builder = ObservationBuilder(
            mj_model=self.mj_model,
            mj_data=self.mj_data,
            heal_ik=self.heal_ik,
            img_size=cfg.get("img_size", 64),
            ft_history_len=cfg.get("ft_history_len", 5),
            camera_name=CAMERA_NAME,
        )

        # ---- Spaces ----
        self.observation_space = spaces.Dict({
            "rgb_d": spaces.Box(0.0, 1.0, shape=(4, 64, 64), dtype=np.float32),
            "ft_history": spaces.Box(-np.inf, np.inf, shape=(5, 6), dtype=np.float32),
            "proprioception": spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float32),
        })
        # 6-D Cartesian delta: [dx, dy, dz, droll, dpitch, dyaw]
        self.action_space = spaces.Box(
            low=np.array([-0.005]*3 + [-0.035]*3, dtype=np.float32),
            high=np.array([ 0.005]*3 + [ 0.035]*3, dtype=np.float32),
            dtype=np.float32,
        )

        # ---- Curriculum ----
        self.curriculum = CurriculumManager(cfg.get("curriculum_config", None))

        # ---- Phase thresholds ----
        phase_cfg = cfg.get("phase", {})
        self.lateral_thresh = phase_cfg.get("lateral_threshold", 0.005)
        self.contact_thresh = phase_cfg.get("contact_threshold", 0.3)

        # ---- Reward weights ----
        self.reward_cfg = cfg.get("reward_weights", {})

        # ---- Renderer ----
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.mj_model, height=480, width=640)

        # ---- Episode bookkeeping ----
        self._step_count = 0
        self._prev_depth = 0.0
        self._phase = 0
        self._had_contact = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_peg_tip(self) -> np.ndarray:
        return self.mj_data.site(PEG_TIP_SITE).xpos.copy()

    def _get_hole_center(self) -> np.ndarray:
        return self.mj_data.body(HOLE_BODY).xpos.copy()

    def _get_ft_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Return F/T readings rotated to world frame."""
        ft_force_raw = self.mj_data.sensor(FT_FORCE_SENSOR).data.copy()
        ft_torque_raw = self.mj_data.sensor(FT_TORQUE_SENSOR).data.copy()
        R_ft = self.mj_data.body(FT_SENSOR_BODY).xmat.reshape(3, 3)
        return R_ft @ ft_force_raw, R_ft @ ft_torque_raw

    def _compute_phase(self) -> int:
        tip = self._get_peg_tip()
        hole = self._get_hole_center()
        lateral_err = np.linalg.norm((tip - hole)[[0, 2]])
        depth = np.dot(tip - hole, INSERTION_AXIS)
        ft_force_world, _ = self._get_ft_world()
        ft_mag = np.linalg.norm(ft_force_world)

        if lateral_err <= self.lateral_thresh and ft_mag > self.contact_thresh:
            return 1   # insertion
        return 0       # approach

    def _peg_y_axis_world(self) -> np.ndarray:
        """Column 1 of the peg body rotation matrix (local Y in world)."""
        R = self.mj_data.body(PEG_BODY).xmat.reshape(3, 3)
        return R[:, 1].copy()

    def _build_info(self) -> dict:
        ft_force_world, ft_torque_world = self._get_ft_world()
        return {
            "peg_tip_world": self._get_peg_tip(),
            "hole_center_world": self._get_hole_center(),
            "ft_force_world": ft_force_world,
            "ft_torque_world": ft_torque_world,
            "peg_y_axis_world": self._peg_y_axis_world(),
            "phase": self._phase,
        }

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Apply curriculum noise to initial joint configuration
        stage = self.curriculum.current_stage()
        pos_noise_m = stage.get("pos_noise_mm", 0.0) / 1000.0
        rot_noise_rad = np.deg2rad(stage.get("rot_noise_deg", 0.0))

        if pos_noise_m > 0 or rot_noise_rad > 0:
            noise_pos = self.np_random.uniform(-pos_noise_m, pos_noise_m, size=3)
            noise_rot = self.np_random.uniform(-rot_noise_rad, rot_noise_rad, size=3)
            # Get current EE pose, perturb, solve IK for new initial joints
            q_current = self.mj_data.qpos[:6].copy()
            q_init = self.heal_ik.delta_to_q_target(q_current, noise_pos, noise_rot)
            self.mj_data.qpos[:6] = q_init

        # Gripper closed
        self.mj_data.ctrl[6] = self.gripper_ctrl
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Reset bookkeeping
        self._step_count = 0
        tip = self._get_peg_tip()
        hole = self._get_hole_center()
        self._prev_depth = np.dot(tip - hole, INSERTION_AXIS)
        self._phase = 0
        self._had_contact = False
        self.obs_builder.reset()

        obs = self.obs_builder.build(self._phase)
        info = self._build_info()
        info["achieved_goal"] = self._prev_depth
        info["desired_goal"] = self.curriculum.current_success_depth()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action.astype(np.float64),
                         self.action_space.low, self.action_space.high)

        delta_pos = action[:3]
        delta_euler = action[3:]

        q_current = self.mj_data.qpos[:6].copy()
        dq_current = self.mj_data.qvel[:6].copy()

        # PyRoKi IK — choose mode based on phase
        if self._phase == 0 and self.cfg.get("ik_pos_only_phase0", True):
            q_target = self.heal_ik.delta_to_q_target(
                q_current, delta_pos, delta_euler, pos_only=True
            )
        else:
            q_target = self.heal_ik.delta_to_q_target(
                q_current, delta_pos, delta_euler, pos_only=False
            )

        # PD torque control → MuJoCo actuators
        torques = self.kp * (q_target - q_current) - self.kd * dq_current
        torques = np.clip(torques, self.ctrl_lo, self.ctrl_hi)
        self.mj_data.ctrl[:6] = torques
        self.mj_data.ctrl[6] = self.gripper_ctrl

        # Step simulation (decimation)
        for _ in range(self.decimation):
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Update phase
        self._phase = self._compute_phase()

        # Check contact
        ft_force_world, _ = self._get_ft_world()
        if np.linalg.norm(ft_force_world) > self.contact_thresh:
            self._had_contact = True

        # Compute reward
        info = self._build_info()
        reward = compute_reward(
            info=info,
            prev_depth=self._prev_depth,
            phase=self._phase,
            cfg=self.reward_cfg,
        )

        # Update depth
        tip = self._get_peg_tip()
        hole = self._get_hole_center()
        depth = np.dot(tip - hole, INSERTION_AXIS)
        self._prev_depth = depth

        # Termination
        self._step_count += 1
        success_depth = self.curriculum.current_success_depth()
        success = depth >= success_depth
        truncated = self._step_count >= self.max_steps
        terminated = success

        # Build obs
        obs = self.obs_builder.build(self._phase)

        # Info for HER and logging
        info["achieved_goal"] = depth
        info["desired_goal"] = success_depth
        info["is_success"] = success
        info["insertion_depth"] = depth
        info["lateral_force"] = np.linalg.norm(ft_force_world[[0, 2]])
        info["had_contact"] = self._had_contact
        info["phase"] = self._phase
        info["step"] = self._step_count

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is not None:
            self._renderer.update_scene(self.mj_data)
            return self._renderer.render()
        return None

    def close(self):
        self.obs_builder.close()
        if self._renderer is not None:
            self._renderer.close()
