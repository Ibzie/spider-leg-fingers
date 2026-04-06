# Spider Leg Finger Experiment

Benchmarking spider leg kinematics against human finger models for robotic manipulation.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YCB objects
python scripts/download_data.py

# Train SLFM on power grasp task
python scripts/train_rl.py model=slfm_5finger task=power_grasp

# Run benchmarks
python scripts/run_benchmarks.py
```

## Project Structure

- `src/models/` - Robot kinematics and dynamics
- `src/physics/` - MuJoCo environment wrappers
- `src/rl/` - PPO agent and training loop
- `src/tasks/` - Manipulation tasks (T1-T6)
- `src/benchmarks/` - Performance metrics (C1-C6, M1-M6)
- `configs/` - Hydra configuration files

## Hardware Requirements

- GPU: CUDA-capable, 8GB VRAM (RTX 4060 class)
- RAM: 16GB
- Storage: 10GB for data and checkpoints

## Citation

```bibtex
@misc{spider_leg_fingers_2026,
  title={Spider-Leg Kinematics as a Computationally Efficient Finger Proxy},
  author={Ibrahim Akhtar},
  year={2026}
}
```
