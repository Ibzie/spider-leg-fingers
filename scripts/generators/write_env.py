import subprocess

env_code = """
import numpy as np
import mujoco
import os
from typing import Dict, Tuple, Optional


class ManipulationEnv:
    \"\"\"MuJoCo environment for manipulation tasks\"\"\"
    
    def __init__(self, model_path: str, task_name: str = "grasp"):
        self.model_path = model_path
        self.task_name = task_name
        
        # Load model
        if os.path.exists(model_path):
            self.model = mujoco.MjModel.from_xml_path(model_path)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.data = mujoco.MjData(self.model)
        
        # Get indices
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.nsensordata = self.model.nsensordata
        
        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.max_steps = 500  # 1 second at 500Hz
        self.current_step = 0
        
        # Observation and action dimensions
        self.obs_dim = self.nq + self.nv + self.nsensordata + 3  # pos + vel + sensors + target
        self.action_dim = self.nu
        
        # Target object
        self.target_pos = np.array([0.05, 0.0, 0.05])
        self.object_id = None
        
    def reset(self) -> np.ndarray:
        \"\"\"Reset environment\"\"\"
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # Randomize initial pose slightly
        self.data.qpos[:7] += np.random.randn(7) * 0.01  # Palm pose
        self.data.qpos[7:] = np.random.randn(self.nq - 7) * 0.1  # Joint angles
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        \"\"\"Take simulation step\"\"\"
        # Apply action (position targets)
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0) * 90  # Scale to degrees
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()
        
        return obs, reward, done, info
    
    def _get_obs(self) -> np.ndarray:
        \"\"\"Get observation\"\"\"
        obs = np.concatenate([
            self.data.qpos[:self.nq],
            self.data.qvel[:self.nv],
            self.data.sensordata[:] if self.nsensordata > 0 else np.array([]),
            self.target_pos
        ])
        return obs.astype(np.float32)
    
    def _compute_reward(self) -> float:
        \"\"\"Compute task reward\"\"\"
        # Get fingertip positions
        reward = 0.0
        
        # Proximity to target
        palm_pos = self.data.qpos[:3]
        dist_to_target = np.linalg.norm(palm_pos - self.target_pos)
        reward += -dist_to_target * 0.1
        
        # Penalize large velocities
        reward += -np.sum(self.data.qvel**2) * 0.001
        
        # Reward for contacts
        if self.nsensordata > 0:
            contact_forces = self.data.sensordata
            reward += np.sum(contact_forces) * 0.01
        
        return reward
    
    def _check_done(self) -> bool:
        \"\"\"Check if episode is done\"\"\"
        if self.current_step >= self.max_steps:
            return True
        
        # Check if palm fell too far
        if self.data.qpos[2] < 0.05:
            return True
            
        return False
    
    def _get_info(self) -> Dict:
        \"\"\"Get additional info\"\"\"
        return {
            'step': self.current_step,
            'palm_height': self.data.qpos[2],
        }


class PowerGraspEnv(ManipulationEnv):
    \"\"\"Task T1: Power grasp of cylinder\"\"\"
    def __init__(self, model_path: str):
        super().__init__(model_path, "power_grasp")
        self.cylinder_radius = 0.02
        self.cylinder_height = 0.08
        
    def _compute_reward(self) -> float:
        # Reward for enclosing object
        reward = super()._compute_reward()
        
        # Add bonus for stable grasp
        if self.nsensordata > 0:
            contacts = self.data.sensordata
            n_contacts = np.sum(contacts > 0.1)
            if n_contacts >= 3:  # At least 3 fingers touching
                reward += 10.0
        
        return reward


class PrecisionPinchEnv(ManipulationEnv):
    \"\"\"Task T2: Precision pinch of cube\"\"\"
    def __init__(self, model_path: str):
        super().__init__(model_path, "precision_pinch")
        self.cube_size = 0.01
        self.max_steps = 1000


def create_env(model_type: str = "slfm", task: str = "power_grasp"):
    \"\"\"Factory function to create environment\"\"\"
    base_path = "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/assets/models"
    
    if model_type == "slfm":
        model_path = os.path.join(base_path, "slfm_5finger.xml")
    elif model_type == "hfm":
        model_path = os.path.join(base_path, "hfm_5finger.xml")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if task == "power_grasp":
        return PowerGraspEnv(model_path)
    elif task == "precision_pinch":
        return PrecisionPinchEnv(model_path)
    else:
        return ManipulationEnv(model_path, task)


if __name__ == "__main__":
    # Test environment
    env = create_env("slfm", "power_grasp")
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action dim: {env.action_dim}")
    
    # Run a few steps
    for i in range(10):
        action = np.random.randn(env.action_dim) * 0.1
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={done}")
        if done:
            break
    
    print("Environment test passed!")
"""

with open(
    "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/src/physics/mujoco_env.py",
    "w",
) as f:
    f.write(env_code.strip())

print("Created mujoco_env.py")
