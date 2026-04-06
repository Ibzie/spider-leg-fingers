#!/usr/bin/env python3
"""
Visualize trained model in MuJoCo viewer
"""

import sys

sys.path.insert(0, "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/src")

import torch
import numpy as np
import mujoco
import mujoco.viewer
import os
import time

from physics.mujoco_env import create_env
from rl.agents.ppo import PPOAgent


def visualize(
    model_type="slfm", task="power_grasp", checkpoint_path=None, n_episodes=5
):
    """Visualize trained model"""

    # Create environment
    env = create_env(model_type, task)

    # Create agent
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device="cpu",  # Use CPU for visualization
    )

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        agent.network.load_state_dict(checkpoint["network_state"])
        print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
        print(
            f"Best reward during training: {checkpoint.get('best_reward', 'unknown')}"
        )
    else:
        print("No checkpoint provided - using random policy")
        if checkpoint_path:
            print(f"(Could not find: {checkpoint_path})")

    print(f"\nVisualizing {model_type.upper()} on {task}")
    print(f"Running {n_episodes} episodes...")
    print("Close the viewer window to stop\n")

    episode = 0
    episode_reward = 0
    episode_step = 0

    # Reset environment
    obs = env.reset()

    # Launch viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and episode < n_episodes:
            step_start = time.time()

            # Get action from policy (deterministic for visualization)
            action, _, _ = agent.select_action(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_step += 1

            # Sync viewer
            viewer.sync()

            # Control simulation speed (real-time ~60 FPS)
            elapsed = time.time() - step_start
            if elapsed < 0.016:  # 60 FPS
                time.sleep(0.016 - elapsed)

            if done:
                print(
                    f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {episode_step}"
                )
                episode += 1
                episode_reward = 0
                episode_step = 0
                obs = env.reset()

    print(f"\nVisualization complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="slfm", choices=["slfm", "hfm"])
    parser.add_argument("--task", type=str, default="power_grasp")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument("--episodes", type=int, default=5)

    args = parser.parse_args()

    # Auto-find checkpoint if not specified
    if args.checkpoint is None:
        save_dir = f"/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/experiments/runs/{args.model}_{args.task}"
        final_model = os.path.join(save_dir, "final_model.pt")
        if os.path.exists(final_model):
            args.checkpoint = final_model
            print(f"Auto-selected checkpoint: {final_model}")

    visualize(args.model, args.task, args.checkpoint, args.episodes)
