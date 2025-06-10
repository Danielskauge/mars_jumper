#!/usr/bin/env python3

"""
Analyze input domain mismatch between training data and Isaac Lab deployment.
This script doesn't require Isaac Lab simulation to run.
"""

import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def load_training_config():
    """Load training configuration to understand preprocessing."""
    config_path = "test_actuators/training_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_normalization_stats():
    """Load normalization statistics from training."""
    stats_path = "test_actuators/normalization_stats.json"
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats

def analyze_training_domain():
    """Analyze what the training data looked like."""
    config = load_training_config()
    stats = load_normalization_stats()
    
    print("=== TRAINING DATA ANALYSIS ===")
    print(f"Training frequency: {config['data']['resampling_frequency_hz']} Hz")
    print(f"Filter cutoff: {config['data']['filter_cutoff_freq_hz']} Hz")
    print(f"Sensor biases: {config['data']['sensor_biases']}")
    print(f"Use residual: {config['model']['use_residual']}")
    print(f"Max torque: {config['data']['max_torque_nm']} Nm")
    
    print("\nInput features (after preprocessing):")
    feature_names = ["angle_rad", "target_angle_rad", "ang_vel_rad", "prev_torque"]
    for i, (name, mean, std) in enumerate(zip(feature_names, stats['input_mean'], stats['input_std'])):
        print(f"  {name:15}: mean={mean:8.4f}, std={std:8.4f}")
    
    print(f"\nTarget (torque): mean={stats['target_mean'][0]:8.4f}, std={stats['target_std'][0]:8.4f}")
    
    return config, stats

def simulate_isaac_lab_inputs():
    """Simulate what Isaac Lab would provide as inputs."""
    print("\n=== ISAAC LAB INPUT SIMULATION ===")
    
    # Simulation parameters
    ANGLE_OFFSET_RAD = math.radians(40.0)  # From run_test.py
    
    # Simulate a constant hold trajectory (like the debug script)
    constant_value = math.pi/8  # Default from run_test_debug.py
    target_angle = constant_value + ANGLE_OFFSET_RAD
    
    print(f"Target angle: {target_angle:.4f} rad ({math.degrees(target_angle):.1f}°)")
    print(f"Initial joint position: {ANGLE_OFFSET_RAD:.4f} rad ({math.degrees(ANGLE_OFFSET_RAD):.1f}°)")
    
    # Simulate what happens during a constant hold
    # Isaac Lab provides raw, unfiltered values
    sim_duration = 5.0  # seconds
    sim_freq = 240.0   # Hz
    num_steps = int(sim_duration * sim_freq)
    
    # Simulate joint motion during constant hold
    # Joint starts at offset, moves toward target
    time_steps = np.linspace(0, sim_duration, num_steps)
    
    # Simple exponential approach to target (like a PD controller would do)
    tau = 0.5  # time constant
    joint_positions = target_angle + (ANGLE_OFFSET_RAD - target_angle) * np.exp(-time_steps / tau)
    
    # Velocity is derivative of position
    joint_velocities = -(ANGLE_OFFSET_RAD - target_angle) / tau * np.exp(-time_steps / tau)
    
    # Target is constant
    target_positions = np.full_like(time_steps, target_angle)
    
    # Previous torque starts at 0, then becomes whatever the actuator outputs
    # For this analysis, assume it settles to some steady-state value
    prev_torques = np.zeros_like(time_steps)
    
    # Stack features as Isaac Lab would
    isaac_lab_features = np.column_stack([
        joint_positions,
        target_positions, 
        joint_velocities,
        prev_torques
    ])
    
    # Analyze statistics
    print("\nIsaac Lab raw inputs (simulated constant hold):")
    feature_names = ["angle_rad", "target_angle_rad", "ang_vel_rad", "prev_torque"]
    for i, name in enumerate(feature_names):
        data = isaac_lab_features[:, i]
        print(f"  {name:15}: mean={np.mean(data):8.4f}, std={np.std(data):8.4f}, range=[{np.min(data):6.3f}, {np.max(data):6.3f}]")
    
    return isaac_lab_features, feature_names

def compare_domains():
    """Compare training vs Isaac Lab input domains."""
    config, stats = analyze_training_domain()
    isaac_features, feature_names = simulate_isaac_lab_inputs()
    
    print("\n=== DOMAIN COMPARISON ===")
    
    # Compare statistics
    training_means = np.array(stats['input_mean'])
    training_stds = np.array(stats['input_std'])
    
    isaac_means = np.mean(isaac_features, axis=0)
    isaac_stds = np.std(isaac_features, axis=0)
    
    print("\nFeature comparison:")
    print(f"{'Feature':<15} {'Training Mean':<12} {'Isaac Mean':<12} {'Diff':<8} {'Training Std':<12} {'Isaac Std':<12} {'Ratio':<8}")
    print("-" * 85)
    
    for i, name in enumerate(feature_names):
        mean_diff = isaac_means[i] - training_means[i]
        std_ratio = isaac_stds[i] / training_stds[i] if training_stds[i] > 0 else float('inf')
        print(f"{name:<15} {training_means[i]:>11.4f} {isaac_means[i]:>11.4f} {mean_diff:>7.3f} {training_stds[i]:>11.4f} {isaac_stds[i]:>11.4f} {std_ratio:>7.2f}")
    
    # Identify major mismatches
    print("\n=== IDENTIFIED ISSUES ===")
    
    issues = []
    
    # Check angle domain
    if abs(isaac_means[0] - training_means[0]) > 0.5:
        issues.append(f"ANGLE DOMAIN MISMATCH: Training centered at {training_means[0]:.3f} rad ({math.degrees(training_means[0]):.1f}°), Isaac at {isaac_means[0]:.3f} rad ({math.degrees(isaac_means[0]):.1f}°)")
    
    # Check target domain  
    if abs(isaac_means[1] - training_means[1]) > 0.5:
        issues.append(f"TARGET DOMAIN MISMATCH: Training centered at {training_means[1]:.3f} rad ({math.degrees(training_means[1]):.1f}°), Isaac at {isaac_means[1]:.3f} rad ({math.degrees(isaac_means[1]):.1f}°)")
    
    # Check velocity scaling
    if isaac_stds[2] / training_stds[2] > 10 or isaac_stds[2] / training_stds[2] < 0.1:
        issues.append(f"VELOCITY SCALING MISMATCH: Training std={training_stds[2]:.4f}, Isaac std={isaac_stds[2]:.4f} (ratio: {isaac_stds[2]/training_stds[2]:.1f})")
    
    # Check if filtering was applied in training
    filter_freq = config['data']['filter_cutoff_freq_hz']
    if filter_freq is not None:
        issues.append(f"FILTERING MISMATCH: Training used {filter_freq} Hz low-pass filter, Isaac Lab uses raw unfiltered data")
    
    # Check sensor biases
    sensor_biases = config['data']['sensor_biases']
    if sensor_biases:
        issues.append(f"SENSOR BIAS MISMATCH: Training corrected biases {sensor_biases}, Isaac Lab uses raw sensor values")
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("No major domain mismatches detected.")
    
    return issues

def suggest_fixes(issues):
    """Suggest fixes for identified issues."""
    print("\n=== SUGGESTED FIXES ===")
    
    if not issues:
        print("No fixes needed - domains appear to match.")
        return
    
    fixes = []
    
    for issue in issues:
        if "ANGLE DOMAIN MISMATCH" in issue:
            fixes.append("Fix angle domain: Ensure Isaac Lab joint positions are in the same range as training data. Check if angle offset is correctly applied.")
        
        if "TARGET DOMAIN MISMATCH" in issue:
            fixes.append("Fix target domain: Ensure target angles in Isaac Lab match the range used during training.")
        
        if "VELOCITY SCALING MISMATCH" in issue:
            fixes.append("Fix velocity scaling: Check if Isaac Lab joint velocities have the same units and scaling as training data.")
        
        if "FILTERING MISMATCH" in issue:
            fixes.append("Fix filtering: Apply the same low-pass filter to Isaac Lab inputs as was used during training, or retrain without filtering.")
        
        if "SENSOR BIAS MISMATCH" in issue:
            fixes.append("Fix sensor biases: Apply the same bias corrections to Isaac Lab inputs as were used during training.")
    
    for i, fix in enumerate(fixes, 1):
        print(f"{i}. {fix}")

def create_visualization():
    """Create visualization of the domain mismatch."""
    config, stats = analyze_training_domain()
    isaac_features, feature_names = simulate_isaac_lab_inputs()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    training_means = np.array(stats['input_mean'])
    training_stds = np.array(stats['input_std'])
    
    for i, (name, ax) in enumerate(zip(feature_names, axes)):
        # Training distribution (normal approximation)
        x_train = np.linspace(training_means[i] - 3*training_stds[i], 
                             training_means[i] + 3*training_stds[i], 100)
        y_train = np.exp(-0.5 * ((x_train - training_means[i]) / training_stds[i])**2)
        y_train /= np.max(y_train)
        
        # Isaac Lab data
        isaac_data = isaac_features[:, i]
        
        ax.plot(x_train, y_train, 'b-', label='Training (normalized)', linewidth=2)
        ax.hist(isaac_data, bins=50, density=True, alpha=0.7, color='red', label='Isaac Lab (simulated)')
        ax.axvline(training_means[i], color='blue', linestyle='--', alpha=0.7, label=f'Training mean: {training_means[i]:.3f}')
        ax.axvline(np.mean(isaac_data), color='red', linestyle='--', alpha=0.7, label=f'Isaac mean: {np.mean(isaac_data):.3f}')
        
        ax.set_title(f'{name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('input_domain_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: input_domain_comparison.png")

def main():
    """Main analysis function."""
    print("Analyzing input domain mismatch between training and Isaac Lab deployment...\n")
    
    try:
        issues = compare_domains()
        suggest_fixes(issues)
        create_visualization()
        
        print(f"\n=== SUMMARY ===")
        if issues:
            print(f"Found {len(issues)} potential issues that could explain the erratic behavior.")
            print("The most likely cause is that Isaac Lab provides raw, unfiltered sensor data")
            print("while the training used filtered and bias-corrected data.")
        else:
            print("No obvious domain mismatches found. The issue may be elsewhere.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 