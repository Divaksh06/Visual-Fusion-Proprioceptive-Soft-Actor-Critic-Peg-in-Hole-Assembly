"""
Assembles all observation modalities from MuJoCo sensors and PyRoKi FK.

Observation Dict
----------------
* rgb_d      : (4, 64, 64) float32  — RGB + normalised depth
* ft_history : (5, 6) float32       — last 5 steps of [fx,fy,fz,tx,ty,tz] world
* proprioception : (20,) float32    — joints, vel, peg tip, orientation, phase
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

FT_FORCE_SENSOR = "ft_force_sensor"
FT_TORQUE_SENSOR = "ft_torque_sensor"
FT_SENSOR_BODY = "force_torque_sensor"
PEG_TIP_SITE = "peg_tip"
MAX_DEPTH_M = 1.5


class ObservationBuilder:
    """Builds the multimodal observation dict each step."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        heal_ik,
        img_size: int = 64,
        ft_history_len: int = 5,
        camera_name: str = "realsense_rgb",
    ):
        self.model = mj_model
        self.data = mj_data
        self.heal_ik = heal_ik
        self.img_size = img_size
        self.ft_history_len = ft_history_len
        self.camera_name = camera_name

        # Renderers for RGB and depth
        self._rgb_renderer = mujoco.Renderer(mj_model, height=img_size, width=img_size)
        self._depth_renderer = mujoco.Renderer(mj_model, height=img_size, width=img_size)

        # F/T history buffer
        self._ft_buffer: deque = deque(maxlen=ft_history_len)

    def close(self):
        """Release GPU resources held by the renderers."""
        if self._rgb_renderer is not None:
            self._rgb_renderer.close()
            self._rgb_renderer = None
        if self._depth_renderer is not None:
            self._depth_renderer.close()
            self._depth_renderer = None

    def reset(self):
        """Clear the F/T history (fill with zeros)."""
        self._ft_buffer.clear()
        for _ in range(self.ft_history_len):
            self._ft_buffer.append(np.zeros(6, dtype=np.float32))

    def _render_rgbd(self) -> np.ndarray:
        """Return (4, H, W) float32 tensor: RGB [0,1] + depth [0,1]."""
        # RGB
        self._rgb_renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self._rgb_renderer.render().copy()  # (H, W, 3) uint8

        # Depth
        self._depth_renderer.update_scene(self.data, camera=self.camera_name)
        self._depth_renderer.enable_depth_rendering()
        depth = self._depth_renderer.render().copy()  # (H, W) float32
        self._depth_renderer.disable_depth_rendering()

        # Normalise
        rgb_norm = rgb.astype(np.float32) / 255.0           # (H, W, 3)
        depth_norm = np.clip(depth / MAX_DEPTH_M, 0, 1)     # (H, W)

        # Channel-first: (4, H, W)
        rgbd = np.concatenate(
            [rgb_norm.transpose(2, 0, 1), depth_norm[np.newaxis]],
            axis=0,
        ).astype(np.float32)
        return rgbd

    def _get_ft_world(self) -> np.ndarray:
        """Read F/T sensor, rotate to world frame, return (6,) vector."""
        force_raw = self.data.sensor(FT_FORCE_SENSOR).data.copy()
        torque_raw = self.data.sensor(FT_TORQUE_SENSOR).data.copy()
        R_ft = self.data.body(FT_SENSOR_BODY).xmat.reshape(3, 3)
        force_w = R_ft @ force_raw
        torque_w = R_ft @ torque_raw
        return np.concatenate([force_w, torque_w]).astype(np.float32)

    def _get_proprioception(self, phase: int) -> np.ndarray:
        """Build (20,) proprioception vector."""
        q = self.data.qpos[:6].copy().astype(np.float32)       # (6,)
        dq = self.data.qvel[:6].copy().astype(np.float32)       # (6,)
        tip_pos = self.data.site(PEG_TIP_SITE).xpos.copy().astype(np.float32)  # (3,)

        # Peg tip orientation via PyRoKi FK (more stable than MuJoCo xmat)
        _, ee_quat = self.heal_ik.fk(q.astype(np.float64))
        ee_quat = np.array(ee_quat, dtype=np.float32)           # (4,) wxyz

        phase_bit = np.array([float(phase)], dtype=np.float32)  # (1,)

        prop = np.concatenate([q, dq, tip_pos, ee_quat, phase_bit])
        assert prop.shape == (20,), f"Proprioception shape mismatch: {prop.shape}"
        return prop

    def build(self, phase: int) -> dict:
        """Return the full observation dict."""
        # RGB-D
        rgbd = self._render_rgbd()

        # F/T
        ft_vec = self._get_ft_world()
        self._ft_buffer.append(ft_vec)
        ft_history = np.stack(list(self._ft_buffer), axis=0).astype(np.float32)  # (5, 6)

        # Proprioception
        prop = self._get_proprioception(phase)

        return {
            "rgb_d": rgbd,
            "ft_history": ft_history,
            "proprioception": prop,
        }
