# Mars Jumper: Lightweight Quadruped with PEAs for Dynamic Jumping


[![Simulation Demo](assets/simulation_demo.gif)]()

## Overview

Mars Jumper is a research project focused on developing a lightweight, low-cost quadruped robot designed for dynamic jumping maneuvers, particularly relevant for exploring challenging terrains like Martian or Lunar lava tubes. This repository contains the simulation environments (built with **Isaac Lab**), reinforcement learning training code, and utilities for deploying control policies. The robot design uniquely incorporates **Parallel Elastic Actuators (PEAs)** in its knees to enhance jumping performance and potentially aid in landing compliance. Notably, this work presents the first deep learning based modeling of PEAs, using neural networks trained on real-world test data to accurately capture their complex dynamics.

## Motivation

Exploring extraterrestrial environments like Mars and the Moon presents unique challenges. While wheeled rovers have been successful, accessing scientifically valuable locations like lava tubes requires navigating complex terrain unsuitable for wheels. Jumping robots offer a promising alternative, leveraging lower gravity to overcome obstacles. However, developing robust jumping capabilities, especially controlling flight and landing, remains difficult and risky to test on expensive hardware.

This project addresses these challenges by:
1.  **Designing a novel, low-cost quadruped:** Making dynamic jump testing more accessible.
2.  **Integrating Parallel Elastic Actuators (PEAs):** Combining servo motors with springs in the knees to potentially boost jump power and absorb landing impacts.
3.  **Using Reinforcement Learning (RL):** Training intelligent controllers (using PPO) in simulation to master complex jumping maneuvers, including takeoff, in-flight attitude control using leg movements, and landing.
4.  **Focusing on Sim2Real Transfer:** Employing techniques like Domain Randomization to ensure policies trained in simulation can effectively control the physical robot.

## Key Features

*   **Lightweight Quadruped Design:** Optimized for agility and dynamic movements. (CAD/design files likely in `robot/`)
*   **Parallel Elastic Actuators (PEAs):** Servo-spring mechanism in the knees for enhanced jump performance. The PEAs are modeled using a Multi-Layer Perceptron (MLP) trained on real-world test data.
*   **Reinforcement Learning Control:** Utilizes Proximal Policy Optimization (PPO) for training robust jumping policies.
*   **Phase-Based Dynamic Jumping:** Focus on single jumps with distinct phases:
    *   **Takeoff:** Achieving desired launch velocity (magnitude and direction).
    *   **Flight:** In-flight attitude control using leg movements as reaction masses.
    *   **Landing:** Stable and compliant touchdown.
*   **Comprehensive Metrics & Error Tracking:** Detailed logging of success rates for each jump phase, peak height/length errors, attitude errors, and command-bucketed performance.
*   **Randomized Jump Commands:** Training with varying target jump heights and lengths to promote robustness.
*   **Simulation Environment:** Developed using **Isaac Lab**, enabling detailed physics simulation and sensor integration (e.g., contact sensors).
*   **Sim2Real Ready:** Incorporates Domain Randomization during training for better transfer to hardware.

## Project Status

**[TODO: Describe current status - e.g., Advanced simulation development in Isaac Lab, PEA modeling complete, RL training for full jump sequence in progress, Hardware prototype undergoing revisions, Initial Sim2Real tests planned/conducted, etc.]**

## Repository Structure

```
mars_jumper/
├── envs/             # Simulation environment definitions
├── robot/            # Robot description files (URDF/MJCF/USD) and CAD models
├── scripts/          # Utility scripts
├── sweeps/           # Hyperparameter sweep configurations (e.g., for WandB)
├── USD_files/        # USD assets for simulation visualization
├── train.py          # Main script for training RL policies
├── play.py           # Script to run and visualize a trained policy
├── play_single_episode.py # Script to run a single episode evaluation
├── play_multiple_episodes.py # Script to run multiple episode evaluations
├── pyproject.toml    # Project dependencies and configuration
├── README.md         # This file
└── ...
```

## Getting Started

### Training

To train a new jumping policy:
```bash
python train.py [arguments...]
```
*   Check `train.py` or use `python train.py --help` for available training arguments (e.g., configuration files, hyperparameters, logging options).
*   Training progress and results are typically logged using Weights & Biases (see `wandb/` directory).

### Evaluation / Visualization

To visualize a trained policy checkpoint:
```bash
python play.py --checkpoint <path_to_checkpoint> [other_arguments...]
```
*   Use `play_single_episode.py` or `play_multiple_episodes.py` for specific evaluation protocols. Check scripts for relevant arguments.

*(Please update with actual citation details)*

## License
MIT

## USD
The URDF in the USD folder is not standalone functional. It must be put with the meshes in the CAD repo to work. It is there because it is the only part of the URDF that needs to be edited, and it is therefore convenient to have it in this repo. 