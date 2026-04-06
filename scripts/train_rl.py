#!/usr/bin/env python3
"""
Simplified training script that works reliably
"""

import sys

sys.path.insert(0, "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/src")

import torch
import numpy as np
import time
import os
from collections import deque

from physics.mujoco_env import create_env
from rl.agents.ppo import PPOAgent


def train(model_type="slfm", task="power_grasp", total_steps=100000, device="cpu"):
    """Main training loop - simplified"""

    print(f"Starting training: {model_type} on {task}")
    print(f"Device: {device}")
    print(f"Total steps: {total_steps:,}")
    print("Press Ctrl+C to stop early\n")

    # Create environment
    env = create_env(model_type, task)

    # Create agent
    agent = PPOAgent(
        obs_dim=env.obs_dim, action_dim=env.action_dim, lr=3e-4, device=device
    )

    # Tracking
    episode_rewards = []
    episode_lengths = []
    best_reward = float("-inf")

    save_dir = f"/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/experiments/runs/{model_type}_{task}"
    os.makedirs(save_dir, exist_ok=True)

    # Training parameters
    rollout_steps = 2048
    n_updates = total_steps // rollout_steps
    ent_coef = 0.05  # Increased from default 0.01 for more exploration

    step = 0
    total_episodes = 0
    start_time = time.time()

    try:
        for update in range(n_updates):
            # Collect rollout
            obs_list, action_list, reward_list, done_list = [], [], [], []
            log_prob_list, value_list = [], []

            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            batch_episodes = 0
            batch_rewards = []

            for _ in range(rollout_steps):
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)

                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(reward)
                done_list.append(done)
                log_prob_list.append(log_prob)
                value_list.append(value)

                episode_reward += reward
                episode_length += 1
                obs = next_obs

                if done:
                    batch_rewards.append(episode_reward)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    total_episodes += 1
                    batch_episodes += 1

                    if episode_reward > best_reward:
                        best_reward = episode_reward

                    # Reset for new episode
                    obs = env.reset()
                    episode_reward = 0
                    episode_length = 0

            # Get final value for GAE
            _, _, next_value = agent.select_action(obs)

            # Compute GAE
            rewards = np.array(reward_list)
            values = np.array(value_list)
            dones = np.array(done_list)
            advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

            # Update policy
            rollout_data = (
                np.array(obs_list),
                np.array(action_list),
                np.array(log_prob_list),
                advantages,
                returns,
            )
            agent.update(rollout_data, epochs=4, ent_coef=ent_coef)

            step += rollout_steps
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed

            # Print progress
            if batch_episodes > 0:
                mean_reward = np.mean(batch_rewards)
                print(
                    f"Update {update + 1}/{n_updates} | Step {step:,} | "
                    f"Episodes: {total_episodes} | "
                    f"Reward: {mean_reward:.2f} | "
                    f"Best: {best_reward:.2f} | "
                    f"Speed: {steps_per_sec:.0f} steps/s"
                )
            else:
                print(
                    f"Update {update + 1}/{n_updates} | Step {step:,} | "
                    f"No episodes completed | Speed: {steps_per_sec:.0f} steps/s"
                )

            # Save periodically
            if update % 10 == 0 and update > 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_{step}.pt")
                torch.save(
                    {
                        "network_state": agent.network.state_dict(),
                        "optimizer_state": agent.optimizer.state_dict(),
                        "step": step,
                        "episode": total_episodes,
                        "best_reward": best_reward,
                    },
                    checkpoint_path,
                )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Final save
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save(
        {
            "network_state": agent.network.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "step": step,
            "episode": total_episodes,
            "best_reward": best_reward,
        },
        final_path,
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(
        f"Total steps: {step:,} | Episodes: {total_episodes} | Time: {elapsed / 60:.1f}m"
    )
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final model: {final_path}")
    print(f"{'=' * 70}\n")

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="slfm", choices=["slfm", "hfm"])
    parser.add_argument("--task", type=str, default="power_grasp")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    train(args.model, args.task, args.steps, args.device)
