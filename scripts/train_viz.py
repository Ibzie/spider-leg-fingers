#!/usr/bin/env python3
"""Training with real-time MuJoCo visualization"""

import sys

sys.path.insert(0, "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/src")

import torch
import numpy as np
import time
import os
import argparse
import mujoco.viewer

from physics.mujoco_env import create_env
from rl.agents.ppo import PPOAgent


def train_with_viz(
    model_type="slfm",
    task="power_grasp",
    total_steps=100000,
    device="cpu",
    visualize=True,
):
    print(f"Starting training: {model_type} on {task}")
    print(f"Device: {device} | Visualize: {visualize}")
    print(f"Total steps: {total_steps:,}")
    print("Press Ctrl+C to stop early\n")

    env = create_env(model_type, task)
    agent = PPOAgent(
        obs_dim=env.obs_dim, action_dim=env.action_dim, lr=3e-4, device=device
    )

    best_reward = float("-inf")
    save_dir = f"/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/experiments/runs/{model_type}_{task}"
    os.makedirs(save_dir, exist_ok=True)

    rollout_steps = 2048
    n_updates = total_steps // rollout_steps
    ent_coef = 0.05

    step = 0
    total_episodes = 0
    start_time = time.time()
    episode_reward = 0
    episode_length = 0
    batch_rewards = []

    if visualize:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            obs = env.reset()

            for update in range(n_updates):
                obs_list, action_list, reward_list, done_list = [], [], [], []
                log_prob_list, value_list = [], []

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

                    viewer.sync()
                    time.sleep(0.001)

                    if done:
                        batch_rewards.append(episode_reward)
                        if episode_reward > best_reward:
                            best_reward = episode_reward
                        total_episodes += 1
                        obs = env.reset()
                        episode_reward = 0
                        episode_length = 0

                        mean_reward = (
                            np.mean(batch_rewards[-10:]) if batch_rewards else 0
                        )
                        print(
                            f"Ep {total_episodes} | R: {episode_reward:.0f} | Best: {best_reward:.0f} | Mean10: {mean_reward:.0f}"
                        )

                _, _, next_value = agent.select_action(obs)
                rewards = np.array(reward_list)
                values = np.array(value_list)
                dones = np.array(done_list)
                advantages, returns = agent.compute_gae(
                    rewards, values, dones, next_value
                )

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
                print(
                    f"Update {update + 1}/{n_updates} | Step {step:,} | Best: {best_reward:.0f} | {step / elapsed:.0f} steps/s"
                )

                if update % 50 == 0:
                    torch.save(
                        {
                            "network_state": agent.network.state_dict(),
                            "step": step,
                            "best_reward": best_reward,
                        },
                        os.path.join(save_dir, f"checkpoint_{step}.pt"),
                    )

    else:
        obs = env.reset()

        for update in range(n_updates):
            obs_list, action_list, reward_list, done_list = [], [], [], []
            log_prob_list, value_list = [], []

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
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                    total_episodes += 1
                    obs = env.reset()
                    episode_reward = 0
                    episode_length = 0

            _, _, next_value = agent.select_action(obs)
            rewards = np.array(reward_list)
            values = np.array(value_list)
            dones = np.array(done_list)
            advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

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

            if batch_rewards:
                mean_reward = np.mean(batch_rewards)
                print(
                    f"Update {update + 1}/{n_updates} | Step {step:,} | Reward: {mean_reward:.0f} | Best: {best_reward:.0f} | {step / elapsed:.0f} steps/s"
                )

            if update % 50 == 0:
                torch.save(
                    {
                        "network_state": agent.network.state_dict(),
                        "step": step,
                        "best_reward": best_reward,
                    },
                    os.path.join(save_dir, f"checkpoint_{step}.pt"),
                )

    torch.save(
        {
            "network_state": agent.network.state_dict(),
            "step": step,
            "best_reward": best_reward,
        },
        os.path.join(save_dir, "final_model.pt"),
    )

    print(f"\nDone! Best: {best_reward:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="slfm")
    parser.add_argument("--task", type=str, default="power_grasp")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--viz", action="store_true", help="Show MuJoCo visualization")
    args = parser.parse_args()

    train_with_viz(args.model, args.task, args.steps, args.device, args.viz)
