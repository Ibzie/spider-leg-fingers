# Spider Leg Finger Experiment

Benchmarking spider leg kinematics against human finger models for robotic manipulation.

## Preliminary Results (T1: Power Grasp)

Training was limited to ~3 hours (~14M steps) due to hardware constraints. Further training expected to improve results significantly.

| Metric | Result | Target | Notes |
|--------|--------|--------|-------|
| **M1 - Success Rate** | 11% | ≥70% | Requires more training |
| **M2 - Grasp Stability** | 0.12 | ≥0.8 | GSI approximation |
| **M4 - Slip Events** | 0/100 | Report | No detected slips |
| **M5 - Time-to-Grasp** | 457ms | Report | Avg when successful |

**Episode Statistics:**
- Mean Reward: 2153 ± 1757
- Max Reward: 8144
- Training Best: 14,229

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train SLFM on power grasp task
python scripts/train_viz.py --model slfm --task power_grasp --steps 100000000 --device cuda --viz

# Run benchmark evaluation
python scripts/benchmark.py --checkpoint experiments/runs/slfm_power_grasp/final_model.pt --episodes 100

# Visualize trained model
python scripts/visualize.py --checkpoint experiments/runs/slfm_power_grasp/final_model.pt --episodes 10
```

## Project Structure

- `src/models/` - Robot kinematics and dynamics
- `src/physics/` - MuJoCo environment wrappers
- `src/rl/` - PPO agent and training loop
- `scripts/` - Training, visualization, and benchmark scripts
- `assets/models/` - MuJoCo XML model definitions

## Task Suite

| Task | Description | Status |
|------|-------------|--------|
| T1 | Power grasp (cylinder) | In Progress |
| T2 | Precision pinch (cube) | Pending |
| T3 | Irregular wrap (mustard) | Pending |
| T4 | Dynamic catch | Pending |
| T5 | In-hand reorientation | Pending |
| T6 | Micromanipulation (peg-in-hole) | Pending |

## Hardware Requirements

- GPU: CUDA-capable, 8GB VRAM (RTX 4060 class)
- RAM: 16GB
- Storage: 10GB for data and checkpoints

## Limitations

Current results are from limited training time (~3 hours / 14M steps). The PDF research proposal targets 100M+ steps for proper convergence. Hardware constraints (single RTX 4060) limited training duration.

## Citation

```bibtex
@misc{spider_leg_fingers_2026,
  title={Spider-Leg Kinematics as a Computationally Efficient Finger Proxy},
  author={Ibrahim Akhtar},
  year={2026}
}
```