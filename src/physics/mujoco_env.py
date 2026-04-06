import numpy as np
import mujoco
import os
from typing import Dict, Tuple, Optional


class ManipulationEnv:
    """MuJoCo environment for manipulation tasks"""

    def __init__(self, model_path: str, task_name: str = "grasp"):
        self.model_path = model_path
        self.task_name = task_name

        if os.path.exists(model_path):
            self.model = mujoco.MjModel.from_xml_path(model_path)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.data = mujoco.MjData(self.model)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.nsensordata = self.model.nsensordata

        self.dt = self.model.opt.timestep
        self.max_steps = 1000
        self.current_step = 0

        self.obs_dim = self.nq + self.nv + self.nsensordata + 3
        self.action_dim = self.nu

        self.action_scale = 1.5  # Reduced from 5.0 for stability

        self.object_id = None
        self.object_init_pos = None
        self.fingertip_site_ids = []

        self._init_task_specifics()

    def _init_task_specifics(self):
        palm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        if palm_body_id >= 0:
            self.palm_body_id = palm_body_id
        else:
            self.palm_body_id = 0

        object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder"
        )
        if object_body_id >= 0:
            self.object_id = object_body_id
            self.object_qposadr = self.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_root"
                )
            ]
        else:
            self.object_id = None
            self.object_qposadr = None

        fingertip_names = [
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
            "finger0_tip",
            "finger1_tip",
            "finger2_tip",
            "finger3_tip",
            "finger4_tip",
            "finger5_tip",
        ]
        self.fingertip_site_ids = []
        for name in fingertip_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                self.fingertip_site_ids.append(site_id)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Check if palm has a freejoint (mobile palm) or is fixed
        palm_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "palm_root"
        )
        if palm_joint_id >= 0:
            palm_qposadr = self.model.jnt_qposadr[palm_joint_id]
            self.data.qpos[palm_qposadr : palm_qposadr + 3] = np.array(
                [
                    np.random.uniform(-0.02, 0.02),
                    np.random.uniform(-0.02, 0.02),
                    np.random.uniform(0.12, 0.18),
                ]
            )
            theta = np.random.uniform(-0.1, 0.1)
            self.data.qpos[palm_qposadr + 3 : palm_qposadr + 7] = np.array(
                [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
            )
            joint_start = palm_qposadr + 7
        else:
            joint_start = 0

        if self.object_id is not None and self.object_qposadr is not None:
            self.data.qpos[self.object_qposadr : self.object_qposadr + 3] = np.array(
                [
                    0.05 + np.random.uniform(-0.01, 0.01),
                    np.random.uniform(-0.01, 0.01),
                    0.04,
                ]
            )
            self.data.qpos[self.object_qposadr + 3 : self.object_qposadr + 7] = (
                np.array([1, 0, 0, 0])
            )
            self.object_init_pos = self.data.qpos[
                self.object_qposadr : self.object_qposadr + 3
            ].copy()
            joint_start = self.object_qposadr + 7

        n_joints = self.nq - joint_start
        if n_joints > 0:
            # Initialize joints to smaller range for stability
            self.data.qpos[joint_start:] = np.random.uniform(-0.05, 0.1, size=n_joints)

        self.data.qvel[:] = np.random.randn(self.nv) * 0.01

        mujoco.mj_forward(self.model, self.data)

        for _ in range(20):
            self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        scaled_action = np.clip(action, -1.0, 1.0) * self.action_scale
        self.data.ctrl[:] = scaled_action

        try:
            mujoco.mj_step(self.model, self.data)
        except Exception as e:
            print(f"Simulation step failed, resetting: {e}")
            obs = self.reset()
            info = {"error": str(e), "terminated": True, "truncated": False}
            return obs, -10.0, True, info

        if not np.all(np.isfinite(self.data.qpos)) or not np.all(
            np.isfinite(self.data.qvel)
        ):
            print("Warning:NaN/Inf detected, resetting environment")
            obs = self.reset()
            info = {"error": "nan_inf", "terminated": True, "truncated": False}
            return obs, -10.0, True, info

        self.current_step += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated = self._check_done()

        done = terminated or truncated

        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated

        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        qpos = np.clip(self.data.qpos[: self.nq], -10, 10)
        qvel = np.clip(self.data.qvel[: self.nv], -50, 50)

        object_pose = np.zeros(3)
        if self.object_id is not None and self.object_qposadr is not None:
            object_pose = self.data.qpos[self.object_qposadr : self.object_qposadr + 3]

        obs = np.concatenate(
            [
                qpos,
                qvel,
                self.data.sensordata[:] if self.nsensordata > 0 else np.array([]),
                object_pose,
            ]
        )
        return obs.astype(np.float32)

    def _get_fingertip_positions(self) -> np.ndarray:
        positions = []
        for site_id in self.fingertip_site_ids:
            pos = self.data.site_xpos[site_id].copy()
            positions.append(pos)
        return np.array(positions)

    def _get_object_position(self) -> np.ndarray:
        if self.object_id is not None:
            return self.data.xpos[self.object_id].copy()
        return np.zeros(3)

    def _get_object_velocity(self) -> np.ndarray:
        if self.object_id is not None and self.object_qposadr is not None:
            object_qveladr = self.model.jnt_dofadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_root"
                )
            ]
            return self.data.qvel[object_qveladr : object_qveladr + 6].copy()
        return np.zeros(6)

    def _get_contact_count(self) -> int:
        if self.nsensordata == 0:
            return 0
        contact_forces = self.data.sensordata[: self.nsensordata]
        return int(np.sum(contact_forces > 0.1))

    def _compute_gsi_approx(self) -> float:
        if self.nsensordata == 0:
            return 0.0
        contact_forces = self.data.sensordata[: self.nsensordata]
        total_force = np.sum(np.abs(contact_forces))
        n_contacts = np.sum(contact_forces > 0.1)

        if n_contacts < 2:
            return 0.0

        force_balance = 1.0 - np.std(contact_forces[contact_forces > 0.1]) / (
            np.mean(contact_forces[contact_forces > 0.1]) + 1e-6
        )
        force_balance = max(0.0, min(1.0, force_balance))

        contact_score = min(1.0, n_contacts / 3.0)
        return float(force_balance * contact_score)

    def _compute_reward(self) -> float:
        return 0.0

    def _check_done(self) -> Tuple[bool, bool]:
        if self.current_step >= self.max_steps:
            return False, True

        # Check for NaN/Inf in state
        if not np.all(np.isfinite(self.data.qpos)):
            return True, False

        # Check if palm has freejoint (mobile)
        palm_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "palm_root"
        )
        if palm_joint_id >= 0:
            palm_qposadr = self.model.jnt_qposadr[palm_joint_id]
            palm_height = self.data.qpos[palm_qposadr + 2]
            if palm_height < 0.02:
                return True, False

        # Check if object fell off
        if self.object_id is not None:
            obj_height = self._get_object_position()[2]
            if obj_height < -0.1:  # Object fell below ground
                return True, False

        return False, False

    def _get_info(self) -> Dict:
        return {
            "step": self.current_step,
            "palm_height": self.data.qpos[2] if self.nq > 2 else 0.0,
            "object_height": self._get_object_position()[2] if self.object_id else 0.0,
            "contact_count": self._get_contact_count(),
            "gsi": self._compute_gsi_approx(),
        }


class PowerGraspEnv(ManipulationEnv):
    """Task T1: Power grasp of cylinder - Lift 500g, hold stable"""

    def __init__(self, model_path: str):
        super().__init__(model_path, "power_grasp")
        self.lift_threshold = 0.06
        self.target_height = 0.10
        self.hold_duration = 0
        self.prev_contacts = 0
        self.prev_height = 0.04
        self.contact_history = []  # Track which fingers have made contact
        self.max_contacts_reached = 0
        self.steps_since_contact_loss = 0

    def reset(self) -> np.ndarray:
        self.hold_duration = 0
        self.prev_contacts = 0
        self.prev_height = 0.04
        self.contact_history = []
        self.max_contacts_reached = 0
        self.steps_since_contact_loss = 0
        return super().reset()

    def _compute_reward(self) -> float:
        reward = 0.0

        if not np.all(np.isfinite(self.data.qpos)) or not np.all(
            np.isfinite(self.data.qvel)
        ):
            return -10.0

        qvel = np.clip(self.data.qvel, -50, 50)
        vel_penalty = np.sum(qvel**2) * 0.001
        reward -= vel_penalty

        object_pos = self._get_object_position()
        object_vel = self._get_object_velocity()
        fingertip_positions = self._get_fingertip_positions()
        n_contacts = self._get_contact_count()
        object_height = object_pos[2]

        # Track contact persistence
        if n_contacts > self.max_contacts_reached:
            reward += (n_contacts - self.max_contacts_reached) * 2.0
            self.max_contacts_reached = n_contacts
            self.steps_since_contact_loss = 0

        # Penalty for losing contacts
        if n_contacts < self.prev_contacts and n_contacts > 0:
            contacts_lost = self.prev_contacts - n_contacts
            reward -= contacts_lost * 3.0
        elif n_contacts == 0 and self.prev_contacts > 0:
            reward -= 5.0

        # Sustained contact bonus
        if n_contacts >= 3:
            reward += n_contacts * 0.3

        self.prev_contacts = n_contacts

        ground_height = 0.04
        lift_threshold = 0.06

        if len(fingertip_positions) > 0 and self.object_id is not None:
            distances = np.linalg.norm(fingertip_positions - object_pos, axis=1)
            min_dist = np.min(distances)
            avg_dist = np.mean(distances)

            reward += np.exp(-min_dist * 10) * 1.0
            reward += np.exp(-avg_dist * 3) * 0.5

            n_close = np.sum(distances < 0.05)
            n_very_close = np.sum(distances < 0.02)
            reward += n_close * 0.3
            reward += n_very_close * 0.5

            if len(distances) >= 5 and n_close >= 3:
                reward += 1.0

        if object_height > ground_height:
            lift_progress = min(
                1.0, (object_height - ground_height) / (lift_threshold - ground_height)
            )
            reward += lift_progress * 2.0

        if object_height > lift_threshold and n_contacts >= 2:
            reward += 3.0

            vel_magnitude = np.linalg.norm(object_vel[:3])
            if vel_magnitude < 0.1:
                reward += (0.1 - vel_magnitude) * 3.0

                self.hold_duration += 1
                if self.hold_duration > 50:
                    reward += min(self.hold_duration / 50.0, 3.0)
                if self.hold_duration > 200:
                    reward += 10.0
            else:
                self.hold_duration = max(0, self.hold_duration - 5)
        else:
            self.hold_duration = 0

        return float(reward)

    def _check_done(self) -> Tuple[bool, bool]:
        terminated, truncated = super()._check_done()

        if not terminated and self.hold_duration >= 300:
            return True, False

        if self.object_id is not None:
            obj_pos = self._get_object_position()
            if obj_pos[2] < -0.05:
                return True, False

        return terminated, truncated


class PrecisionPinchEnv(ManipulationEnv):
    """Task T2: Precision pinch of cube"""

    def __init__(self, model_path: str):
        super().__init__(model_path, "precision_pinch")
        self.target_height = 0.08

    def _compute_reward(self) -> float:
        reward = 0.0

        object_pos = self._get_object_position()

        if self.nsensordata > 0:
            contact_forces = self.data.sensordata[: self.nsensordata]
            n_contacts = np.sum(contact_forces > 0.1)

            if n_contacts >= 2:
                reward += 1.0
            elif n_contacts == 1:
                reward += 0.3

        object_height = object_pos[2]
        if object_height > 0.05:
            reward += (object_height - 0.05) * 10.0

        return float(reward)

    def _get_info(self) -> Dict:
        info = super()._get_info()
        info["task"] = "precision_pinch"
        return info


def create_env(model_type: str = "slfm", task: str = "power_grasp"):
    """Factory function to create environment"""
    base_path = (
        "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/assets/models"
    )

    if task == "power_grasp":
        model_file = "slfm_powergrasp_v2.xml"
    elif model_type == "slfm":
        model_file = "slfm_5finger.xml"
    elif model_type == "hfm":
        model_file = "hfm_5finger.xml"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_path = os.path.join(base_path, model_file)

    if task == "power_grasp":
        return PowerGraspEnv(model_path)
    elif task == "precision_pinch":
        return PrecisionPinchEnv(model_path)
    else:
        return ManipulationEnv(model_path, task)


if __name__ == "__main__":
    env = create_env("slfm", "power_grasp")
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action dim: {env.action_dim}")
    print(f"Object ID: {env.object_id}")
    print(f"Fingertip sites: {len(env.fingertip_site_ids)}")

    for i in range(10):
        action = np.random.randn(env.action_dim) * 0.1
        obs, reward, done, info = env.step(action)
        print(
            f"Step {i}: reward={reward:.3f}, obj_height={info['object_height']:.3f}, contacts={info['contact_count']}, done={done}"
        )
        if done:
            break

    print("Environment test passed!")
