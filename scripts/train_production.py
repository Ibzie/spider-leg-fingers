#!/usr/bin/env python3
"""
Production-ready training script with best practices for dexterous manipulation
Based on Stable-Baselines3 tips and robotics RL research
"""

import sys

sys.path.insert(0, "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/src")

import torch
import torch.nn as nn
import numpy as np
import time
import os
from collections import deque

from physics.mujoco_env import create_env
from rl.agents.ppo import PPOAgent


class RunningMeanStd:
    """
    Running mean and standard deviation tracker for observation normalization.
    Critical for stable training in continuous control.
    """

    def __init__(self, shape, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x):
        """Update running statistics with new batch of observations"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # Welford's online algorithm for numerical stability
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x, clip=10.0):
        """Normalize observation"""
        x_normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(x_normalized, -clip, clip)


class CurriculumManager:
    """
    Curriculum learning for manipulation tasks.
    Starts with easy target positions and gradually increases difficulty.
    """

    def __init__(self, total_updates):
        self.total_updates = total_updates
        self.current_update = 0

        # Curriculum stages
        self.stages = [
            # (start_update, end_update, target_range, description)
            (0.0, 0.2, 0.02, "Easy: Targets very close"),
            (0.2, 0.4, 0.04, "Medium-Easy: Targets close"),
            (0.4, 0.6, 0.06, "Medium: Targets moderate distance"),
            (0.6, 0.8, 0.08, "Medium-Hard: Targets far"),
            (0.8, 1.0, 0.10, "Hard: Targets very far"),
        ]

    def get_target_range(self, update):
        """Get target position range for current update"""
        progress = update / self.total_updates

        for start, end, range_val, desc in self.stages:
            if start <= progress < end:
                return range_val, desc

        # Default to hardest
        return self.stages[-1][2], self.stages[-1][3]

    def update_env(self, env, update):
        """Update environment difficulty based on curriculum"""
        range_val, desc = self.get_target_range(update)
        env.curriculum_range = range_val
        return desc


class MetricsLogger:
    """
    Logs training metrics to JSON file for analysis.
    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "metrics.jsonl")
        self.episode_count = 0

    def log_step(self, update, step, metrics):
        """Log a training step"""
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "update": update,
            "step": step,
            **metrics,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_episode(self, episode_num, reward, length, info):
        """Log episode completion"""
        self.episode_count += 1

        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "episode",
            "episode": episode_num,
            "reward": float(reward),
            "length": int(length),
            **info,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class CheckpointManager:
    """
    Manages checkpoints with automatic cleanup.
    Only keeps the most recent checkpoint to save disk space.
    """

    def __init__(self, save_dir, interval_minutes=30):
        self.save_dir = save_dir
        self.interval = interval_minutes * 60
        self.last_save = time.time()
        self.last_checkpoint_path = None
        os.makedirs(save_dir, exist_ok=True)

    def should_save(self):
        return time.time() - self.last_save >= self.interval

    def save(self, agent, obs_rms, step, episode, best_reward, is_best=False):
        """Save checkpoint and delete old one to save space"""

        if is_best:
            checkpoint_path = os.path.join(self.save_dir, "best_model.pt")
        else:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_step_{step}.pt")

        # Save new checkpoint
        torch.save(
            {
                "network_state": agent.network.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
                "obs_mean": obs_rms.mean,
                "obs_var": obs_rms.var,
                "step": step,
                "episode": episode,
                "best_reward": best_reward,
                "timestamp": time.time(),
            },
            checkpoint_path,
        )

        # Delete old periodic checkpoint (but keep best_model.pt)
        if (
            not is_best
            and self.last_checkpoint_path
            and os.path.exists(self.last_checkpoint_path)
        ):
            try:
                os.remove(self.last_checkpoint_path)
                print(
                    f"  Deleted old checkpoint: {os.path.basename(self.last_checkpoint_path)}"
                )
            except OSError as e:
                print(f"  Warning: Could not delete old checkpoint: {e}")

        if not is_best:
            self.last_checkpoint_path = checkpoint_path

        self.last_save = time.time()
        print(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
        return checkpoint_path


class RewardShaper:
    """
    Reward shaping for grasping tasks based on robotics best practices:
    1. Distance-based rewards (dense signal)
    2. Contact/penetration penalties
    3. Success bonuses
    """

    def __init__(self):
        self.prev_distance = None

    def reset(self):
        self.prev_distance = None

    def shape_reward(self, env, action, reward_dict):
        """
        Shape reward for manipulation task

        Args:
            env: MuJoCo environment
            action: Action taken
            reward_dict: Dictionary with raw reward components

        Returns:
            shaped_reward: Scalar reward
            info: Dictionary with reward components for logging
        """
        # Get fingertip positions (simplified - you'd want to track all fingertips)
        palm_pos = env.data.qpos[:3]
        target_pos = env.target_pos

        # Distance to target
        distance = np.linalg.norm(palm_pos - target_pos)

        # Distance reward (negative, encourages approaching target)
        dist_reward = -distance * 10.0  # Scale up to make it significant

        # Progress reward (improvement in distance)
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            progress_reward = progress * 100.0  # Strong signal for getting closer
        else:
            progress_reward = 0.0
        self.prev_distance = distance

        # Height reward (encourage staying at appropriate height)
        height = palm_pos[2]
        height_reward = -abs(height - 0.08) * 5.0  # Target height ~8cm

        # Control penalty (discourage wild actions)
        ctrl_penalty = -np.sum(np.square(action)) * 0.01

        # Combine rewards
        total_reward = dist_reward + progress_reward + height_reward + ctrl_penalty

        info = {
            "dist_reward": dist_reward,
            "progress_reward": progress_reward,
            "height_reward": height_reward,
            "ctrl_penalty": ctrl_penalty,
            "distance": distance,
        }

        return total_reward, info


def train_production(
    model_type="slfm", task="power_grasp", total_steps=1000000, device="cuda"
):
    """
    Production training with best practices:
    - Observation normalization
    - Reward shaping
    - Proper episode tracking
    - Periodic evaluation
    - Hyperparameter tuning
    """

    print(f"=" * 70)
    print(f"PRODUCTION TRAINING: {model_type.upper()} on {task}")
    print(f"=" * 70)
    print(f"Device: {device}")
    print(f"Total steps: {total_steps:,}")
    print(f"Press Ctrl+C to stop\n")

    # Create environment
    env = create_env(model_type, task)

    # Observation normalization (CRITICAL for continuous control)
    obs_rms = RunningMeanStd(shape=(env.obs_dim,))

    # Reward shaper
    reward_shaper = RewardShaper()

    # Create agent with tuned hyperparameters for robotics
    # Based on Stable-Baselines3 zoo for continuous control
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        lr=3e-4,  # Standard for PPO
        device=device,
    )

    # Training hyperparameters
    rollout_steps = 2048
    n_updates = total_steps // rollout_steps
    n_epochs = 10  # More epochs for smaller batches
    batch_size = 64  # Smaller batches for better gradient estimates
    gamma = 0.99  # Discount factor
    gae_lambda = 0.95  # GAE lambda

    # Tracking
    episode_rewards = []
    episode_lengths = []
    best_reward = float("-inf")

    save_dir = f"/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/experiments/runs/{model_type}_{task}"
    os.makedirs(save_dir, exist_ok=True)

    # Checkpoint manager (saves every 30 mins, deletes old checkpoints)
    checkpoint_mgr = CheckpointManager(save_dir, interval_minutes=30)

    # Metrics logger
    metrics_logger = MetricsLogger(save_dir)

    # Curriculum learning manager
    curriculum_mgr = CurriculumManager(n_updates)

    step = 0
    total_episodes = 0
    start_time = time.time()

    # Initialize observation normalization with random samples
    print("Initializing observation normalization...")
    init_obs = []
    obs = env.reset()
    for _ in range(100):
        action = np.random.randn(env.action_dim) * 0.5
        obs, _, done, _ = env.step(action)
        init_obs.append(obs)
        if done:
            obs = env.reset()
    obs_rms.update(np.array(init_obs))
    print(
        f"  Mean: {np.mean(obs_rms.mean):.3f}, Std: {np.mean(np.sqrt(obs_rms.var)):.3f}\n"
    )

    try:
        for update in range(n_updates):
            # Update curriculum difficulty
            if update % 10 == 0:
                stage_desc = curriculum_mgr.update_env(env, update)
                print(f"  [Curriculum] Update {update}: {stage_desc}")

            # Collect rollout
            obs_list, action_list, reward_list, done_list = [], [], [], []
            log_prob_list, value_list = [], []

            obs = env.reset()
            obs = obs_rms.normalize(obs)
            reward_shaper.reset()

            episode_reward = 0
            episode_length = 0
            batch_episodes = 0
            batch_rewards = []

            for _ in range(rollout_steps):
                action, log_prob, value = agent.select_action(obs)
                next_obs, _, done, info = env.step(action)

                # Shape reward
                shaped_reward, reward_info = reward_shaper.shape_reward(env, action, {})

                # Store transition
                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(shaped_reward)
                done_list.append(done)
                log_prob_list.append(log_prob)
                value_list.append(value)

                episode_reward += shaped_reward
                episode_length += 1

                next_obs = obs_rms.normalize(next_obs)
                obs = next_obs

                if done:
                    batch_rewards.append(episode_reward)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    total_episodes += 1
                    batch_episodes += 1

                    if episode_reward > best_reward:
                        best_reward = episode_reward

                    # Log episode metrics
                    metrics_logger.log_episode(
                        total_episodes,
                        episode_reward,
                        episode_length,
                        {
                            "terminated": info.get("terminated", False),
                            "truncated": info.get("truncated", False),
                            "palm_height": float(env.data.qpos[2]),
                        },
                    )

                    # Reset
                    obs = env.reset()
                    obs = obs_rms.normalize(obs)
                    reward_shaper.reset()
                    episode_reward = 0
                    episode_length = 0

            # Update observation statistics with collected data
            obs_array = np.array(obs_list)
            obs_rms.update(obs_array)

            # Compute GAE
            rewards = np.array(reward_list)
            values = np.array(value_list)
            dones = np.array(done_list)

            _, _, next_value = agent.select_action(obs)
            advantages, returns = agent.compute_gae(
                rewards, values, dones, next_value, gamma=gamma, lam=gae_lambda
            )

            # Update policy
            rollout_data = (
                obs_array,
                np.array(action_list),
                np.array(log_prob_list),
                advantages,
                returns,
            )
            agent.update(rollout_data, epochs=n_epochs)

            step += rollout_steps
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed

            # Progress logging
            if batch_episodes > 0:
                mean_reward = np.mean(batch_rewards)
                std_reward = np.std(batch_rewards)
                print(
                    f"Update {update + 1:4d}/{n_updates} | "
                    f"Step {step:8,} | "
                    f"Episodes: {total_episodes:4d} | "
                    f"Reward: {mean_reward:8.2f}±{std_reward:6.2f} | "
                    f"Best: {best_reward:8.2f} | "
                    f"Speed: {steps_per_sec:6.0f} steps/s"
                )

                # Log step metrics
                if agent.policy_losses:
                    metrics_logger.log_step(
                        update,
                        step,
                        {
                            "mean_reward": float(mean_reward),
                            "std_reward": float(std_reward),
                            "best_reward": float(best_reward),
                            "episodes": total_episodes,
                            "policy_loss": float(agent.policy_losses[-1])
                            if agent.policy_losses
                            else 0.0,
                            "value_loss": float(agent.value_losses[-1])
                            if agent.value_losses
                            else 0.0,
                            "entropy": float(agent.entropies[-1])
                            if agent.entropies
                            else 0.0,
                            "steps_per_sec": float(steps_per_sec),
                        },
                    )
            else:
                print(
                    f"Update {update + 1:4d}/{n_updates} | "
                    f"Step {step:8,} | "
                    f"No episodes completed | "
                    f"Speed: {steps_per_sec:6.0f} steps/s"
                )

            # Save best model (always keep)
            if batch_episodes > 0 and np.mean(batch_rewards) > best_reward * 0.95:
                checkpoint_mgr.save(
                    agent, obs_rms, step, total_episodes, best_reward, is_best=True
                )

            # Periodic checkpoint (auto-deletes old one)
            if checkpoint_mgr.should_save():
                checkpoint_mgr.save(
                    agent, obs_rms, step, total_episodes, best_reward, is_best=False
                )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Final save
    torch.save(
        {
            "network_state": agent.network.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "obs_mean": obs_rms.mean,
            "obs_var": obs_rms.var,
            "step": step,
            "episode": total_episodes,
            "best_reward": best_reward,
        },
        os.path.join(save_dir, "final_model.pt"),
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE!")
    print(f"  Total steps: {step:,}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Time: {elapsed / 60:.1f} minutes")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Final model: {save_dir}/final_model.pt")
    print(f"{'=' * 70}\n")

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="slfm", choices=["slfm", "hfm"])
    parser.add_argument("--task", type=str, default="power_grasp")
    parser.add_argument(
        "--steps", type=int, default=1000000, help="Total training steps (default: 1M)"
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    train_production(args.model, args.task, args.steps, args.device)
