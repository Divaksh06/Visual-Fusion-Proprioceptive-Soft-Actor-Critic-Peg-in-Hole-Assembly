# **VFP-SAC: Visual + Force Based RL for Peg-in-Hole**

This project is my implementation of a **peg-in-hole robotic assembly task** using **Reinforcement Learning**, mainly based on **Soft Actor Critic (SAC)** and **Hindsight Experience Replay (HER)**.

The goal is to train a **UR5 robot** (in PyBullet) to insert a cylindrical peg into a hole, even with noise, misalignment, and contact forces.  
I combine **vision (RGB-D)** + **force-torque data** + **proprioception**, and fuse them using a simple **cross-modal attention** module.

This README explains the whole system in a simple way.

---

## **1. Why This Project?**

Peg-in-hole looks easy but it is surprisingly hard because:

- You need **sub-millimeter precision**
- Even 1–2 degrees of rotation error causes jamming
- Vision gets noisy near contact
- Force alone cannot tell global alignment
- RL struggles with **sparse rewards**

So I wanted to try a multimodal RL approach to see if it performs better.

---

## **2. What the System Does (Two Phases)**

### **Phase 1 — Vision-based movement**
- The robot uses RGB-D images to move the peg near the hole.
- Only rough alignment is needed here.

### **Phase 2 — Force-based insertion**
- When the peg touches the hole boundary, force/torque signals become more important.
- The robot performs fine corrections and inserts smoothly.

The agent automatically learns when to trust vision and when to trust force.

---

## **3. What Inputs the Agent Gets**

The observation consists of:

- **RGB-D image** (256×256×4)
- **Force/Torque values** (6 numbers × 5 time steps)
- **Joint angles and velocities**
- **End-effector pose**
- **Phase indicator** (0 = vision phase, 1 = force phase)

All of this becomes a **~275-dimensional state vector** after encoding.

---

## **4. Actions**

The agent outputs **6D cartesian velocities**:

[vx, vy, vz, wx, wy, wz]

These control how the robot moves in 3D space.

---

## **5. Algorithms Used**

### ✔ **Soft Actor-Critic (SAC)**
- Good for continuous control.
- More stable than PPO for contact tasks.
- Encourages exploration using entropy.

### ✔ **Hindsight Experience Replay (HER)**
- Makes learning possible even when rewards are sparse.
- If the robot fails to insert, HER treats whatever depth it reached as a “goal” and learns from it.

### ✔ **Curriculum Learning**
Training starts easy and becomes harder:
- Big clearance → small clearance  
- Small noise → high noise  
- No rotation → ±6° rotation errors  

This improves training stability.

---

## **6. Simulation Setup**

- Simulator: **PyBullet**
- Robot: **UR5 with Robotiq 85 gripper**
- Peg radius: **18 mm**
- Hole radius: **20 mm**
- Episode length: **500 steps**
- Training time: **~24–36 hours** on an RTX 3060
- Automatic curriculum progression

---

## **7. Folder Structure**

project/

│── config/

│── envs/ # PyBullet environment

│── models/ # Encoders & attention fusion

│── algorithms/ # SAC, HER, Curriculum

│── utils/

│── train.py

└── test.py

---

## **8. Training**

Run with GUI: python train.py --cuda --render

Faster (headless): python train.py --cuda


---

## **9. Testing**
python test.py --checkpoint <path_to_saved_model>

---

## **10. What I Learned**

- Vision alone is unreliable near contact.
- Force alone can’t guide long-range alignment.
- Combining both using attention improves stability.
- SAC is better than PPO for this task.
- HER is basically necessary for sparse rewards.
- Curriculum learning helps a lot in robotics tasks.

---

## **12. Limitations / Future Work**

- Real force sensors are much noisier than simulation.
- Could use domain randomization for sim-to-real transfer.
- Vision encoder could be lighter or replaced by a ViT.
- Tactile sensors could make insertion smoother.

---

## **13. Conclusion**

This project shows that **multimodal reinforcement learning** can solve precise robotic assembly tasks like peg-in-hole with high accuracy. Using SAC + HER + cross-modal fusion + curriculum learning, the robot learns to use **vision when far** and **force when close**, achieving **sub-millimeter insertion accuracy**.
