# Peg-in-Hole Robotic Insertion — Multimodal RL System

A reinforcement learning system for precision peg-in-hole assembly using the
HEAL robot arm. The agent learns to insert a cylindrical peg into a fixed
hole on a table by fusing **RGB-D vision**, **force/torque sensing**, and
**proprioception** through a learned gating module. Training uses **Soft
Actor-Critic (SAC)** augmented with **Hindsight Experience Replay (HER)** and
an **adaptive curriculum**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Scene Geometry](#scene-geometry)
3. [Architecture & Pipeline](#architecture--pipeline)
4. [File Structure](#file-structure)
5. [Code Walkthrough](#code-walkthrough)
6. [Installation](#installation)
7. [Running the Code](#running-the-code)
8. [Configuration](#configuration)
9. [Critical Implementation Notes](#critical-implementation-notes)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

The task is **peg-in-hole insertion** — a fundamental robotic assembly
primitive. A 6-DOF HEAL robot arm with a Robotiq-85 gripper holds a
cylindrical peg and must insert it into a hole fixture on the table.

The hole assembly is **rotated** so its opening faces the **+Y world axis**
(not downward). This means the insertion direction is +Y, and the peg axis
must be aligned with +Y at insertion time.

Two back-ends cooperate at every step:

| Back-end | Role |
|----------|------|
| **MuJoCo** (`heal5.xml`) | Physics simulation, collision, rendering, F/T sensors |
| **PyRoKi** (`heal.urdf`) | Forward / inverse kinematics with joint-limit handling |

The policy outputs 6-D Cartesian deltas (3 translation + 3 rotation) which
are converted to joint targets via PyRoKi IK, then tracked by a PD torque
controller sent to MuJoCo actuators.

---

## Scene Geometry

Understanding the coordinate frames is critical — mistakes here cause the peg
to miss by centimeters.

| Element | Value |
|---------|-------|
| World frame origin | `(0, 0, 0)` |
| Robot base | `pos=(-0.10, 0.0, 0.425)` `euler=(0, 0, 0)` |
| Table top | `pos=(0.0, 0.0, 0.425)` — z-surface at `z=0.425` |
| Hole assembly | `pos=(0.5, 0.0, 0.425)` `euler=(-1.571, 0, 0)` |
| Insertion direction | **+Y world frame** |

- `euler=(-1.571, 0, 0)` rotates the hole so its axis points in the **+Y direction** of the world frame.
- **Peg body:** child of `robotiq_85_base_link`, `local pos=(0, 0, 0.22)`, `euler=(1.5708, 0, 0)`. The `tcp_link` / `peg_tip_link` in the URDF coincides with the peg tip site.
- **Peg tip site:** always use `mj_data.site("peg_tip").xpos` — never the body center.
- **FT sensor site:** `ft_sensor_site` on body `force_torque_sensor` (between `end_effector` and `robotiq_85_base_link`).
- **Camera:** `realsense_rgb` on `camera_sensor_link`, `fovy=60°`, attached near the wrist, `euler=(0, 3.14, -1.57)`.
- **Gripper:** Robotiq 85, single actuated joint (`finger_joint`, ctrl index 6). Equality constraints mirror all other finger joints. Gripper stays **CLOSED** during insertion.

> **Key geometric fact:** The hole assembly is rotated so its opening faces +Y.
> Your approach trajectory must account for this — the peg axis must be aligned
> with the +Y world axis at insertion time, **not -Z** (the naive default for a
> vertically-mounted arm).

---

## Architecture & Pipeline

```
            ┌──────────────┐
            │  MuJoCo Sim  │
            │  heal5.xml   │
            └──────┬───────┘
                   │ sensors + physics
      ┌────────────┼────────────────┐
      ▼            ▼                ▼
┌──────────┐ ┌───────────┐  ┌──────────────┐
│ RGB-D    │ │  F/T      │  │ Propriocep-  │
│ Camera   │ │  Sensor   │  │ tion (joints │
│(64×64×4) │ │(5×6 hist) │  │ + tip + quat │
└────┬─────┘ └─────┬─────┘  │ + phase_bit) │
     │             │        └──────┬───────┘
     ▼             ▼               │
┌───────────┐ ┌───────────┐        │
│  Visual   │ │  Force    │        │
│  Encoder  │ │  Encoder  │        │
│ CNN→256   │ │ 1DCNN→128 │        │
└─────┬─────┘ └─────┬─────┘        │
      │             │              │
      ▼             ▼              │
      ┌──────────────────────┐     │
      │    Gating Module     │◄────┘ (phase_bit)
      │ Sigmoid gates [g_v,  │
      │ g_f] → fused (256)   │
      └──────────┬───────────┘
                 │
                 ▼             ▼
      ┌────────────────────────────────┐
      │         Actor Network          │
      │ concat(fused, proprioception)  │
      │ → MLP → mean + log_std         │
      │ → tanh-squashed Gaussian       │
      │ → 6-D delta action             │
      └──────────┬─────────────────────┘
                 │
                 ▼
      ┌────────────────────┐
      │    PyRoKi IK       │
      │ delta → q_target   │
      └──────────┬─────────┘
                 │
                 ▼
      ┌────────────────────┐
      │  PD Torque Control │
      │  → mj_data.ctrl    │
      └──────────┬─────────┘
                 │
                 ▼
      ┌────────────────────┐
      │  mujoco.mj_step()  │
      │  (×5 decimation)   │
      └────────────────────┘
```

**Training Algorithm:**

The agent trains with SAC (automatic entropy tuning) combined with a
contact-aware HER buffer that only relabels episodes that experienced real
physical contact. An adaptive curriculum progressively increases task
difficulty across four stages (from zero noise / 5 mm depth to full noise /
15 mm depth), with regression protection that steps back a stage if the
success rate drops too far.

The critic is **asymmetric** — it receives the privileged ground-truth offset
between the peg tip and hole centre during training (directly from MuJoCo
state). The actor never sees this, ensuring it learns from sensor data only.

---

## File Structure

```
peg_in_hole/
├── config/
│   ├── default.yaml               # All hyperparameters
│   ├── curriculum_stages.yaml     # Curriculum stage definitions
│   └── assets/
│       ├── heal5.xml              # MuJoCo scene (user-provided)
│       └── heal.urdf              # URDF for PyRoKi (user-provided)
├── peg_in_hole/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── peg_hole_env.py        # Main MuJoCo Gymnasium environment
│   │   ├── reward.py              # Modular reward function
│   │   ├── observation_builder.py # Assembles RGB-D, F/T, proprioception
│   │   └── curriculum.py         # Stage manager + progression logic
│   ├── kinematics/
│   │   ├── __init__.py
│   │   └── pyroki_ik.py          # PyRoKi wrapper: FK, IK, delta→joints
│   ├── models/
│   │   ├── __init__.py
│   │   ├── visual_encoder.py     # CNN for RGB-D frames
│   │   ├── force_encoder.py      # 1-D CNN for F/T history
│   │   ├── gating_module.py      # Learned soft gating
│   │   ├── actor.py              # Phase-conditioned SAC actor
│   │   └── critic.py             # Asymmetric twin critic
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── sac.py                # SAC with automatic entropy tuning
│   │   ├── her_buffer.py         # HER replay + contact-aware relabeling
│   │   └── curriculum_sac.py    # SAC + curriculum stage coordination
│   └── utils/
│       ├── __init__.py
│       ├── logger.py             # WandB / console logging
│       ├── normalizer.py         # Running mean/std per obs branch
│       └── checkpoint.py        # Save / load model weights
├── train.py                      # Main training entry point
├── test.py                       # Evaluation / testing script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Code Walkthrough

### `kinematics/pyroki_ik.py`

Wraps PyRoKi's JAX-based Levenberg-Marquardt IK solver behind a
numpy-in/numpy-out interface. Three key methods:

- **`fk(q)`** — Forward kinematics returning EE position and wxyz quaternion.
- **`ik(target_pos, target_quat, q_init, pos_only)`** — Full IK solve with
  warm-starting from the current configuration. When `pos_only=True`
  (approach phase), orientation weight is set to zero for faster, more stable
  convergence.
- **`delta_to_q_target(q_current, delta_pos, delta_euler)`** — The bridge
  between the policy's Cartesian delta output and joint-space targets for the
  PD controller. Computes the current EE pose via FK, applies the delta, and
  solves IK for the resulting target.

### `envs/peg_hole_env.py`

Gymnasium-compatible environment. On each `step()`:

1. Clip the action to physical limits (±5 mm translation, ±2° rotation).
2. Convert the Cartesian delta to a joint target via `HealIK.delta_to_q_target()`.
3. Compute PD torques and write to `mj_data.ctrl[:6]`.
4. Step MuJoCo 5 times (control decimation: policy at 100 Hz, sim at 500 Hz).
5. Read sensors: peg tip position, F/T (rotated to world frame), camera.
6. Determine phase (approach vs insertion) from lateral error + contact force.
7. Compute reward (all 6 components).
8. Return observation dict, reward, terminated, truncated, info.

**MuJoCo Name Constants:**

```python
# Bodies
PEG_BODY        = "peg"
HOLE_BODY       = "hole_assembly"
FT_SENSOR_BODY  = "force_torque_sensor"

# Sites
PEG_TIP_SITE    = "peg_tip"          # always use this for peg tip position
FT_SITE         = "ft_sensor_site"   # used internally by MuJoCo sensors

# Sensors (read via mj_data.sensor(...).data)
FT_FORCE_SENSOR  = "ft_force_sensor"   # shape (3,) [fx, fy, fz] in sensor frame
FT_TORQUE_SENSOR = "ft_torque_sensor"  # shape (3,) [tx, ty, tz] in sensor frame

# Camera
CAMERA_NAME = "realsense_rgb"  # fovy=60, on camera_sensor_link, euler=(0, 3.14, -1.57)

# Actuators (ctrl array index order)
# [0]=turret, [1]=shoulder, [2]=elbow,
# [3]=wrist_1, [4]=wrist_2, [5]=wrist_3, [6]=gripper_motor

# Insertion geometry
INSERTION_AXIS = np.array([0., 1., 0.])        # +Y world frame
HOLE_WORLD_POS = np.array([0.5, 0.0, 0.425])   # read from sim at reset
```

**Syncing PyRoKi and MuJoCo:**

```python
# At every environment step:
q_current  = mj_data.qpos[:6]   # joint_1..joint_6 from MuJoCo
dq_current = mj_data.qvel[:6]

# Policy outputs: 6D Cartesian delta
delta_pos   = action[:3]   # meters,  clipped to ±0.005
delta_euler = action[3:]   # radians, clipped to ±0.035 (~2°)

# PyRoKi: delta → joint target
q_target = heal_ik.delta_to_q_target(q_current, delta_pos, delta_euler)

# PD control → torques → MuJoCo actuators
KP = np.array([80., 80., 80., 40., 40., 40.])
KD = np.array([10., 10., 10.,  5.,  5.,  5.])
torques = KP * (q_target - q_current) - KD * dq_current
torques = np.clip(torques, [-100,-100,-100,-50,-50,-50],
                           [ 100, 100, 100, 50, 50, 50])
mj_data.ctrl[:6] = torques
mj_data.ctrl[6]  = 50.0   # gripper closed throughout task
mujoco.mj_step(mj_model, mj_data)
```

**Phase Logic:**

```python
lateral_err = norm((tip - hole_ctr)[[0, 2]])   # X and Z error only
depth       = dot(tip - hole_ctr, [0,1,0])     # +Y projection
ft_mag      = norm(rotate_to_world(ft_force))

# Phase 0 — Approach:  lateral_err > 0.005m OR depth < 0.0
# Phase 1 — Insertion: lateral_err <= 0.005m AND ft_mag > 0.3 N
```

`phase_bit` is included in proprioception and fed to actor + gating module.

### `envs/reward.py`

Six additive reward components:

| Component | Active in | Purpose |
|-----------|-----------|---------|
| `R_approach` | Phase 0 only | Penalise lateral distance to hole (XZ plane) |
| `R_align` | Both phases | Penalise peg axis misalignment with +Y |
| `R_insertion` | Phase 1 only | Reward forward depth progress |
| `R_force` | Phase 1 only | Penalise excessive lateral contact force |
| `R_success` | Both (triggered) | Large bonus when full depth reached |
| `R_step` | Both | Small constant penalty for speed |

> **Critical:** `R_approach` must be exactly `0.0` in phase 1. If it stays
> active, the agent finds a local optimum of hovering at the hole entrance
> (high approach reward, zero insertion risk).

```python
def compute_reward(info, prev_depth, phase):
    lateral_err = np.linalg.norm((peg_tip - hole_ctr)[[0, 2]])
    depth       = np.dot(peg_tip - hole_ctr, ins_axis)
    delta_depth = depth - prev_depth
    lat_force   = np.linalg.norm(ft_world[[0, 2]])
    axis_align  = np.dot(peg_y_world, ins_axis)   # 1.0 = perfect alignment

    R_approach  = -alpha * lateral_err              if phase == 0 else 0.0
    R_align     = -beta  * (1.0 - axis_align)       # both phases
    R_insertion =  gamma * max(0.0, delta_depth)    if phase == 1 else 0.0
    R_force     = -delta * max(0.0, lat_force - force_limit) if phase == 1 else 0.0
    R_success   = S_bonus if depth >= success_depth else 0.0
    R_step      = -epsilon

    return R_approach + R_align + R_insertion + R_force + R_success + R_step
```

Starting weights (`config/default.yaml → reward_weights`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `alpha` | 5.0 | Approach distance weight |
| `beta` | 2.0 | Axis alignment weight |
| `gamma` | 10.0 | Insertion depth progress weight |
| `delta` | 1.0 | Lateral force penalty weight |
| `force_limit` | 5.0 N | |
| `success_bonus` | 200.0 | |
| `success_depth` | 0.015 m | 15 mm full insertion |
| `epsilon` | 0.01 | Step penalty |

### `models/visual_encoder.py`

```
Input:  (B, 4, 64, 64)  — channels 0–2: RGB [0,1]; channel 3: depth/1.5m
Arch:   Conv(4→16, 5×5, stride=2) → ReLU
        Conv(16→32, 3×3, stride=2) → ReLU
        Conv(32→64, 3×3, stride=2) → ReLU
        AdaptiveAvgPool(4×4)
        Linear(64×4×4 → 256) → LayerNorm → ReLU
Output: (B, 256)
```
Use **LayerNorm**, NOT BatchNorm.

### `models/force_encoder.py`

```
Input:  (B, 5, 6)  — last 5 timesteps of [fx,fy,fz,tx,ty,tz], world frame
Arch:   Conv1d(6→32, kernel=3, padding=1) → ReLU
        Conv1d(32→64, kernel=3, padding=1) → ReLU
        Flatten → Linear(64×5 → 128) → LayerNorm → ReLU
Output: (B, 128)
```
The `fy` channel (insertion axis) dominates during phase 1. Temporal conv
captures the contact ramp-up signature.

### `models/gating_module.py` — Learned Soft Gating

```
Input:  visual_feat (B, 256), force_feat (B, 128), phase_bit (B, 1)
Arch:
  g_in = concat([visual_feat, force_feat, phase_bit])  → (B, 385)
  Linear(385→128) → ReLU → Linear(128→2) → Sigmoid → [g_v, g_f] ∈ [0,1]²
  v_proj = Linear(256→256)(visual_feat)
  f_proj = Linear(128→256)(force_feat)
  fused  = g_v * v_proj + g_f * f_proj
Output: (B, 256)
```

Use **Sigmoid**, NOT Softmax. Sigmoid allows both gates to suppress
simultaneously (e.g., camera occluded AND sensor noisy). Softmax forces
zero-sum competition.

Expected learned behaviour:
- **Phase 0:** `g_v ~ 0.7–0.9`, `g_f ~ 0.1–0.3`
- **Phase 1:** `g_v ~ 0.2–0.4`, `g_f ~ 0.6–0.9`

Log both gate values every eval step. If they don't shift between phases,
check that `phase_bit` is not zeroed by the observation normaliser.

### `models/actor.py`

```
Input:  fused (B, 256), proprioception (B, 20)
Arch:
  x = concat([fused, proprioception])  → (B, 276)
  Linear(276→256) → LayerNorm → ReLU
  Linear(256→256) → LayerNorm → ReLU
  Linear(256→12)  → (mean_6, log_std_6)
Action: [dx, dy, dz, droll, dpitch, dyaw]
  translation: ±5 mm  |  rotation: ±2° (0.035 rad)
Sampling: reparameterisation, tanh squash, log_prob correction
log_std clamped: [-5, 2]
```

### `models/critic.py` — Asymmetric Twin Critics

```python
# Privileged input (training only):
true_offset = mj_data.site("peg_tip").xpos - mj_data.body("hole_assembly").xpos

# Two critics Q1, Q2 (clipped double-Q):
x_train = concat([fused, prop, action, true_offset])  → (B, 285)
x_eval  = concat([fused, prop, action])               → (B, 282)
MLP(285→512→256→1)
```

The actor **never** sees `true_offset`. At test time, set `true_offset=zeros`
or retrain the critic head without it.

### `algorithms/sac.py`

- Automatic entropy: `target_entropy = -6` (`-action_dim`)
- Gradient clip: norm 1.0 on all parameters
- Soft update: `tau = 0.005`
- Adam `lr = 3e-4` for actor, critic, and alpha
- Update every environment step

### `algorithms/her_buffer.py` — Contact-Aware HER

- Strategy: `"future"`, `k=4`
- Goal: **scalar `insertion_depth`** (not xyz — xyz goals create degenerate relabeling)
  ```python
  achieved_goal = dot(peg_tip - hole_ctr, [0,1,0])
  desired_goal  = 0.015  # metres
  ```
- **Contact-aware filter:** only relabel transitions from episodes where at
  least one step had `norm(ft_force_world) > 0.3 N`. Episodes with zero
  contact produce meaningless achieved goals.
- Do **NOT** normalise rewards — distorts the relabeled reward scale.

### `algorithms/curriculum_sac.py`

Four stages (`config/curriculum_stages.yaml`):

| Stage | Name | Pos noise | Rot noise | Success depth | Advance threshold |
|-------|------|-----------|-----------|---------------|-------------------|
| 0 | coarse_approach | 0 mm | 0° | 5 mm | 80% |
| 1 | noisy_approach | 2 mm | 2° | 8 mm | 75% |
| 2 | tight_insertion | 4 mm | 4° | 12 mm | 70% |
| 3 | full_task | 5 mm | 6° | 15 mm | 65% |

Check success rate every 100 episodes. If it drops more than 20% below the
current threshold, step back one stage (regression protection).

---

## Installation

### Prerequisites

- Python 3.10+ (required by PyRoKi / JAX)
- A GPU is recommended for training but not strictly required

### Steps

```bash
# 1. Clone or create the project directory
git clone <your-repo-url> peg_in_hole_project
cd peg_in_hole_project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows

# 3. Install PyRoKi (from source — not yet on PyPI as a stable release)
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
cd ..

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Place your robot description files
#    Copy heal5.xml → config/assets/heal5.xml
#    Copy heal.urdf → config/assets/heal.urdf
#
#    The URDF must define peg_tip_link as described above.
#    The XML must define the peg_tip site, hole_assembly body,
#    force_torque_sensor body, ft_force_sensor / ft_torque_sensor sensors,
#    and the realsense_rgb camera.
```

### Verify Installation

```bash
python -c "import mujoco; print('MuJoCo', mujoco.__version__)"
python -c "import pyroki; print('PyRoKi OK')"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

### Requirements

```
mujoco>=3.0.0
gymnasium>=0.29.0
torch>=2.1.0
numpy>=1.24.0
pyyaml>=6.0
pyroki>=0.1.0
scipy>=1.11.0
wandb>=0.16.0
imageio>=2.31.0
```

> **PyRoKi note:** `heal.urdf` has no mesh geometry (links are mass-only),
> so it loads without any STL assets. The URDF is purely kinematic — MuJoCo
> handles all collision and rendering.

---

## Running the Code

Follow this order strictly. Validate geometry and sensing **before** touching RL.

### Step 1 — FK Verification

The training script automatically runs FK verification on startup. To test standalone:

```bash
python -c "
from peg_in_hole.envs.peg_hole_env import PegInHoleEnv
import yaml
cfg = yaml.safe_load(open('config/default.yaml'))
env = PegInHoleEnv(cfg)
print('Environment created successfully')
env.close()
"
```

The verification asserts PyRoKi and MuJoCo agree within 1 mm on a known config:

```python
q_test = np.zeros(6)
pyroki_pos, _ = heal_ik.fk(q_test)
mj_data.qpos[:6] = q_test
mujoco.mj_forward(mj_model, mj_data)
mujoco_pos = mj_data.site("peg_tip").xpos
assert np.allclose(pyroki_pos, mujoco_pos, atol=1e-3), \
    f"FK mismatch: pyroki={pyroki_pos}, mujoco={mujoco_pos}"
```

### Step 2 — PD Control Verification

Verify the arm tracks a fixed Cartesian waypoint before training:

```bash
python -c "
import numpy as np, mujoco, yaml
from peg_in_hole.envs.peg_hole_env import PegInHoleEnv

cfg = yaml.safe_load(open('config/default.yaml'))
env = PegInHoleEnv(cfg, render_mode='human')
obs, info = env.reset()

for _ in range(100):
    action = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, r, term, trunc, info = env.step(action)
    env.render()
    if term or trunc:
        break
print('Peg tip:', info['peg_tip_world'])
env.close()
"
```

### Step 3 — Train

```bash
# Basic training (no rendering — fastest)
python train.py --config config/default.yaml --cuda

# Training with rendering (slower, useful for debugging)
python train.py --config config/default.yaml --cuda --render

# Custom seed
python train.py --config config/default.yaml --cuda --seed 123
```

Training will:
1. Run FK verification (assertion fails if mismatch > 1 mm).
2. Warm up with 5,000 random-action steps.
3. Begin SAC updates with HER relabeling.
4. Progress through curriculum stages as success rate improves.
5. Save checkpoints to `checkpoints/` every 50,000 steps and on new best.
6. Log metrics to WandB (or console if WandB unavailable).

### Step 4 — Test

```bash
python test.py --checkpoint checkpoints/best.pt --render --episodes 50
```

Per-episode output includes insertion depth, success flag, maximum lateral
force, and phase switch step. Aggregated statistics are printed at the end.

---

## Configuration

All hyperparameters live in `config/default.yaml`. Key ones to tune:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `control_decimation` | 5 | Policy at 100 Hz, sim at 500 Hz |
| `episode_length` | 500 | Max steps per episode |
| `reward_weights.alpha` | 5.0 | Approach distance weight |
| `reward_weights.gamma` | 10.0 | Insertion depth progress weight |
| `reward_weights.success_bonus` | 200.0 | Full insertion bonus |
| `phase.contact_threshold` | 0.3 | N — lower if phase never switches |
| `her_k` | 4 | HER future-relabeling count |
| `warmup_steps` | 5000 | Random exploration before updates |
| `total_steps` | 2,000,000 | |
| `eval_every` | 10,000 | |
| `save_every` | 50,000 | |

Curriculum stages are in `config/curriculum_stages.yaml`.

---

## Critical Implementation Notes

### `heal5.xml`-Specific Issues

1. **Insertion axis is +Y, not -Z.**
   `hole_assembly euler="-1.571 0 0"` rotates the hole so its axis is world
   +Y. Every reward distance, IK target, and insertion depth calculation must
   use `dot(vec, [0,1,0])`. Using -Z (vertical default) will drive the peg
   into the table.

2. **`peg_tip` site vs body center.**
   Always read `mj_data.site("peg_tip").xpos`. Never read
   `mj_data.body("peg").xpos` (body center, not tip).

3. **F/T sensor frame rotation.**
   Sensors report in the `force_torque_sensor` body frame, not world frame.
   Rotate before use:
   ```python
   ft_force_raw = mj_data.sensor("ft_force_sensor").data.copy()   # (3,)
   ft_torq_raw  = mj_data.sensor("ft_torque_sensor").data.copy()  # (3,)
   R_ft_world   = mj_data.body("force_torque_sensor").xmat.reshape(3,3)
   ft_force_world  = R_ft_world @ ft_force_raw
   ft_torque_world = R_ft_world @ ft_torq_raw
   ft_vec = np.concatenate([ft_force_world, ft_torque_world])     # (6,)
   ```

4. **Gripper equality constraints.**
   All gripper joints are slaved to `finger_joint` via `<equality>` in the
   XML. Set `mj_data.ctrl[6] = 50.0` only. Never set passive joint positions
   directly.

5. **PyRoKi `joint_5` sign difference.**
   URDF has `rpy="-0.5236 0 3.1416"` while the XML has
   `euler="0.5236 0 3.1416"` — a sign flip. The FK verification assertion
   catches this immediately. When `exclude_last=True` is set on the
   proprioception normaliser, the `phase_bit` is preserved correctly.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Peg drives into table | Using -Z instead of +Y as insertion axis | Set `INSERTION_AXIS = [0, 1, 0]` everywhere |
| Agent hovers at hole entrance | `R_approach` still active in phase 1 | Verify `R_approach = 0.0` when `phase == 1` |
| HER produces no improvement | Using xyz goals | Use scalar `insertion_depth` as HER goal |
| Phase never switches to 1 | Contact threshold too high | Log raw `norm(ft_force_world)` and lower threshold |
| IK diverges near hole | Singular arm config at `(0.5, 0, 0.425)` | Always warm-start with `q_init=q_current` |
| FK assertion fails | `joint_5` sign mismatch URDF vs XML | Fix sign in URDF or add compensation |
| Gate values don't shift between phases | `phase_bit` zeroed by normaliser | Set `exclude_last=True` on proprioception normaliser |

### Five Most Common Failure Modes

1. **Wrong insertion axis** — using -Z instead of +Y. The peg drives into the table. Use `INSERTION_AXIS = [0,1,0]` everywhere.
2. **`R_approach` active in phase 1** — agent hovers at entrance without inserting. Zero `R_approach` completely when `phase == 1`.
3. **HER with xyz goals** — relabeled episodes point to random air positions. Use scalar `insertion_depth` as the goal.
4. **Phase never switching** — contact threshold too high for your sim parameters. Log raw `norm(ft_force_world)` during random rollouts and set threshold to ~10% of typical contact magnitude.
5. **PyRoKi IK diverges near hole** — arm near a singular configuration at `(0.5, 0, 0.425)`. Always warm-start from `q_init=q_current` and add joint regularisation in the IK cost.

---

> **Before running:** You must provide `heal5.xml` and `heal.urdf` in
> `config/assets/`. These are specific to the HEAL robot and are not
> included. PyRoKi must be installed from source. Follow the recommended
> implementation order (Steps 1–4) — validate geometry and PD control before
> starting RL training. Most peg-in-hole failures come from geometry or
> sensing bugs, not the algorithm.