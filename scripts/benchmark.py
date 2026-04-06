#!/usr/bin/env python3
"""Benchmark trained model on manipulation metrics"""

import sys

sys.path.insert(0, "/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/src")

import torch
import numpy as np
import time
import argparse
from physics.mujoco_env import create_env
from rl.agents.ppo import PPOAgent


def run_benchmark(
    checkpoint_path, model_type="slfm", task="power_grasp", n_episodes=100
):
    print("=" * 60)
    print("SLFM Power Grasp Benchmark")
    print("=" * 60)

    env = create_env(model_type, task)
    agent = PPOAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, device="cpu")

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent.network.load_state_dict(checkpoint["network_state"])
        print(f"Loaded checkpoint: {checkpoint_path}")
        if "best_reward" in checkpoint:
            print(f"Training best reward: {checkpoint['best_reward']:.2f}")

    print(f"\nRunning {n_episodes} evaluation episodes...")
    print("-" * 60)

    results = {
        "success": 0,  # M1: Task success rate
        "grasp_stability": [],  # M2: GSI approximation
        "contact_variance": [],  # M3: Contact force variance
        "slip_events": 0,  # M4: Slip events
        "time_to_grasp": [],  # M5: Time-to-grasp
        "episodes": [],
    }

    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_contacts = []
        grasp_start = None
        held_steps = 0
        prev_obj_pos = env._get_object_position().copy()
        slip_detected = False

        for step in range(1000):
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            obj_pos = env._get_object_position()
            obj_height = obj_pos[2]
            n_contacts = info.get("contact_count", 0)
            episode_contacts.append(n_contacts)

            # Detect grasp start (object lifted with contacts)
            if grasp_start is None and obj_height > 0.06 and n_contacts >= 2:
                grasp_start = step

            # Track stability while grasped
            if obj_height > 0.06 and n_contacts >= 2:
                held_steps += 1
                # Detect slip: object moving significantly while contacts exist
                obj_vel = np.linalg.norm(obj_pos - prev_obj_pos)
                if obj_vel > 0.01:
                    slip_detected = True

            prev_obj_pos = obj_pos.copy()

            if done:
                break

        # Episode metrics
        success = held_steps >= 100  # Success if held for 100 steps (0.2s)
        results["success"] += success
        results["grasp_stability"].append(min(held_steps / 100.0, 1.0))
        results["contact_variance"].append(
            np.var(episode_contacts) if episode_contacts else 0
        )
        results["time_to_grasp"].append(grasp_start if grasp_start else 1000)
        results["slip_events"] += 1 if slip_detected else 0

        results["episodes"].append(
            {
                "reward": episode_reward,
                "held_steps": held_steps,
                "grasp_start": grasp_start,
                "success": success,
            }
        )

        if (ep + 1) % 20 == 0:
            print(
                f"Episode {ep + 1}/{n_episodes} | Success: {results['success']}/{ep + 1}"
            )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nM1 - Task Success Rate: {results['success'] / n_episodes * 100:.1f}%")
    print(f"    ({results['success']}/{n_episodes} successful grasps)")

    print(f"\nM2 - Grasp Stability Index: {np.mean(results['grasp_stability']):.2f}")
    print(f"    (avg hold duration ratio)")

    print(f"\nM3 - Contact Force Variance: {np.mean(results['contact_variance']):.2f}")
    print(f"    (lower is more stable)")

    print(f"\nM4 - Slip Events: {results['slip_events']}/{n_episodes}")
    print(f"    ({results['slip_events'] / n_episodes * 100:.1f}% episodes with slip)")

    grasp_times = [t for t in results["time_to_grasp"] if t < 1000]
    if grasp_times:
        print(f"\nM5 - Time-to-Grasp: {np.mean(grasp_times):.0f} steps")
        print(f"    ({np.mean(grasp_times) * 2:.0f}ms at 500Hz)")
    else:
        print(f"\nM5 - Time-to-Grasp: No successful grasps")

    print(f"\nEpisode Rewards:")
    rewards = [e["reward"] for e in results["episodes"]]
    print(f"    Mean: {np.mean(rewards):.1f}")
    print(f"    Std: {np.std(rewards):.1f}")
    print(f"    Max: {np.max(rewards):.1f}")
    print(f"    Min: {np.min(rewards):.1f}")

    print("\n" + "=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/runs/slfm_power_grasp/checkpoint_7395328.pt",
    )
    parser.add_argument("--model", type=str, default="slfm")
    parser.add_argument("--task", type=str, default="power_grasp")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    run_benchmark(args.checkpoint, args.model, args.task, args.episodes)
