"""
PyRoKi IK wrapper for the HEAL arm.

URDF joints used for IK : joint_1 … joint_6  (6-DOF revolute)
End-effector link       : peg_tip_link
                          (child of tcp_link, offset 0.22 m along peg axis
                           from robotiq_85_base_link — coincides with
                           peg_tip site in heal5.xml)

PyRoKi loads from **URDF** (no mesh geometry needed).
MuJoCo remains the physics simulator and sensor provider.
The two are kept in sync at every step.

NOTE: PyRoKi uses JAX + Levenberg–Marquardt internally.  This wrapper
exposes a simple numpy-in / numpy-out interface that the RL env calls.
"""

from __future__ import annotations

import os
import numpy as np
from scipy.spatial.transform import Rotation

# Force JAX to use CPU for IK — avoids CUDA version conflict with PyTorch
# (PyTorch uses CUDA 13, JAX plugin requires CUDA 12).
# IK is a tiny 6-variable LM solve; CPU is faster than GPU here anyway.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import logging as _logging
    _logging.getLogger("jaxls").setLevel(_logging.WARNING)

    import jax
    import jax.numpy as jnp
    import jax_dataclasses as jdc
    import jaxlie
    import jaxls
    import pyroki as pk
    import yourdfpy

    _PYROKI_AVAILABLE = True
except ImportError:
    _PYROKI_AVAILABLE = False

# Defaults — overridden by constructor args
_DEFAULT_URDF = "config/assets/heal.urdf"
_EE_LINK = "peg_tip_link"
_ARM_JOINTS = ["joint_1", "joint_2", "joint_3",
               "joint_4", "joint_5", "joint_6"]
_JOINT_LIMITS_LO = -3.1415
_JOINT_LIMITS_HI = 3.1415


class HealIK:
    """Thin wrapper around PyRoKi for FK and IK of the HEAL arm."""

    def __init__(self, urdf_path: str = _DEFAULT_URDF):
        if not _PYROKI_AVAILABLE:
            raise ImportError(
                "PyRoKi and its dependencies (jax, jaxlie, jaxls, yourdfpy) "
                "must be installed.  pip install pyroki"
            )
        self._urdf = yourdfpy.URDF.load(urdf_path, build_collision_scene_graph=False,
                                         build_scene_graph=True, load_meshes=False)
        self._robot = pk.Robot.from_urdf(self._urdf)
        self._ee_link_name = _EE_LINK
        self._ee_link_idx = self._robot.links.names.index(_EE_LINK)
        self._num_joints = self._robot.joints.num_actuated_joints

        # Cache for warm-starting
        self._last_q = np.zeros(self._num_joints, dtype=np.float64)

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward kinematics.

        Parameters
        ----------
        q : (6,) joint angles for joint_1..joint_6

        Returns
        -------
        ee_pos : (3,) in world frame
        ee_quat : (4,) wxyz quaternion
        """
        q_jax = jnp.array(q, dtype=jnp.float32)
        all_poses = self._robot.forward_kinematics(q_jax)  # (num_links, 7) wxyz_xyz
        pose_7 = np.array(all_poses[self._ee_link_idx])     # (7,) wxyz_xyz
        ee_quat = pose_7[:4].copy()   # wxyz
        ee_pos = pose_7[4:].copy()    # xyz
        return ee_pos, ee_quat

    # ------------------------------------------------------------------
    # Inverse Kinematics
    # ------------------------------------------------------------------
    def ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_init: np.ndarray,
        pos_only: bool = False,
    ) -> np.ndarray:
        """
        Inverse kinematics via PyRoKi's LM solver.

        Parameters
        ----------
        target_pos : (3,) desired peg tip position, world frame
        target_quat : (4,) desired orientation wxyz
        q_init : (6,) warm start (current joint config)
        pos_only : if True, ignore orientation (approach phase)

        Returns
        -------
        q_sol : (6,) joint angles, clipped to joint limits
        """
        target_pos_jax = jnp.array(target_pos, dtype=jnp.float32)
        target_wxyz_jax = jnp.array(target_quat, dtype=jnp.float32)
        target_link_index = jnp.array(self._ee_link_idx)

        # Build a robot with the warm-start as default config
        q_init_jax = jnp.array(q_init, dtype=jnp.float32)
        robot_warmstart = pk.Robot.from_urdf(
            self._urdf, default_joint_cfg=q_init_jax
        )

        joint_var = robot_warmstart.joint_var_cls(0)

        # Build costs
        if pos_only:
            # Position-only cost
            costs = [
                pk.costs.pose_cost_analytic_jac(
                    robot_warmstart,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz_jax), target_pos_jax
                    ),
                    target_link_index,
                    pos_weight=50.0,
                    ori_weight=0.0,  # ignore orientation
                ),
                pk.costs.limit_constraint(robot_warmstart, joint_var),
            ]
        else:
            costs = [
                pk.costs.pose_cost_analytic_jac(
                    robot_warmstart,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz_jax), target_pos_jax
                    ),
                    target_link_index,
                    pos_weight=50.0,
                    ori_weight=10.0,
                ),
                pk.costs.limit_constraint(robot_warmstart, joint_var),
            ]

        sol = (
            jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
            .analyze()
            .solve(
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            )
        )
        q_sol = np.array(sol[joint_var])
        q_sol = np.clip(q_sol, _JOINT_LIMITS_LO, _JOINT_LIMITS_HI)
        self._last_q = q_sol.copy()
        return q_sol

    # ------------------------------------------------------------------
    # Delta-action → joint target
    # ------------------------------------------------------------------
    def delta_to_q_target(
        self,
        q_current: np.ndarray,
        delta_pos: np.ndarray,
        delta_euler: np.ndarray,
        pos_only: bool = False,
    ) -> np.ndarray:
        """
        Convert a Cartesian delta action (policy output) to a joint-space target.

        Parameters
        ----------
        q_current : (6,) current joint angles
        delta_pos : (3,) [dx, dy, dz] metres, world frame
        delta_euler : (3,) [droll, dpitch, dyaw] radians, world frame
        pos_only : pass through to IK

        Returns
        -------
        q_target : (6,) joint angles for PD control
        """
        ee_pos, ee_quat = self.fk(q_current)  # ee_quat is wxyz

        target_pos = ee_pos + delta_pos

        # Orientation update
        dR = Rotation.from_euler("xyz", delta_euler)
        # Convert wxyz → xyzw for scipy
        R_current = Rotation.from_quat(
            [ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]
        )
        R_target = dR * R_current
        target_xyzw = R_target.as_quat()  # xyzw
        target_wxyz = np.array(
            [target_xyzw[3], target_xyzw[0], target_xyzw[1], target_xyzw[2]]
        )

        return self.ik(target_pos, target_wxyz, q_init=q_current, pos_only=pos_only)
