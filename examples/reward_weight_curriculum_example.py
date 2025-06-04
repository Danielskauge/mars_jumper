"""
Example configurations for reward weight curriculum in the jumping robot environment.

This shows different strategies for progressively increasing reward weights based on success metrics.
"""

from isaaclab.managers import CurriculumTermCfg
from terms import curriculums

# Strategy 1: Focus on precision as success improves
precision_curriculum = CurriculumTermCfg(
    func=curriculums.progress_reward_weights_by_metric,
    params={
        "metric_name": "full_jump_success_rate",
        "reward_weight_configs": {
            # Increase attitude control importance as agent gets better
            "attitude_landing": {
                "initial_weight": 0.1,
                "target_weight": 1.0,
                "metric_threshold": 0.8,
                "metric_start": 0.3,
            },
            "attitude_landing_trans": {
                "initial_weight": 30.0,
                "target_weight": 150.0,
                "metric_threshold": 0.8,
                "metric_start": 0.3,
            },
            # Increase landing precision requirements
            "feet_height": {
                "initial_weight": 0.1,
                "target_weight": 2.0,
                "metric_threshold": 0.9,
                "metric_start": 0.4,
            },
            "landing_base_height": {
                "initial_weight": 0.05,
                "target_weight": 0.5,
                "metric_threshold": 0.9,
                "metric_start": 0.4,
            },
        },
        "min_steps_between_updates": 500,
        "smoothing_factor": 0.02,  # Slow, gradual changes
    },
)

# Strategy 2: Progressive refinement - start with basic success, add precision
progressive_refinement_curriculum = CurriculumTermCfg(
    func=curriculums.progress_reward_weights_by_metric,
    params={
        "metric_name": "takeoff_success_rate",  # Start with takeoff success
        "reward_weight_configs": {
            # Basic trajectory rewards start high
            "rel_cmd_error_huber": {
                "initial_weight": 10.0,
                "target_weight": 5.0,  # Reduce as agent improves
                "metric_threshold": 0.7,
                "metric_start": 0.2,
            },
            # Precision rewards start low
            "attitude_landing": {
                "initial_weight": 0.05,
                "target_weight": 0.8,
                "metric_threshold": 0.8,
                "metric_start": 0.5,
            },
        },
        "min_steps_between_updates": 300,
        "smoothing_factor": 0.05,
    },
)

# Strategy 3: Landing-focused curriculum based on flight success
landing_focused_curriculum = CurriculumTermCfg(
    func=curriculums.progress_reward_weights_by_metric,
    params={
        "metric_name": "flight_success_rate",
        "reward_weight_configs": {
            # Only increase landing rewards when flight is working
            "attitude_landing": {
                "initial_weight": 0.1,
                "target_weight": 1.5,
                "metric_threshold": 0.9,
                "metric_start": 0.6,
            },
            "feet_height": {
                "initial_weight": 0.2,
                "target_weight": 3.0,
                "metric_threshold": 0.9,
                "metric_start": 0.7,
            },
            "landing_contact_forces": {
                "initial_weight": 0.01,
                "target_weight": 0.1,
                "metric_threshold": 0.95,
                "metric_start": 0.8,
            },
        },
        "min_steps_between_updates": 400,
        "smoothing_factor": 0.03,
    },
)

# Strategy 4: Multi-phase curriculum - different metrics for different phases
def create_multi_phase_curriculum():
    """Creates multiple curriculum terms for different learning phases."""
    return {
        # Phase 1: Focus on takeoff success
        "takeoff_phase": CurriculumTermCfg(
            func=curriculums.progress_reward_weights_by_metric,
            params={
                "metric_name": "takeoff_success_rate",
                "reward_weight_configs": {
                    "relative_cmd_error_huber_flight_trans": {
                        "initial_weight": 50.0,
                        "target_weight": 200.0,
                        "metric_threshold": 0.8,
                        "metric_start": 0.2,
                    },
                },
                "min_steps_between_updates": 200,
                "smoothing_factor": 0.1,
            },
        ),
        
        # Phase 2: Focus on flight success once takeoff is good
        "flight_phase": CurriculumTermCfg(
            func=curriculums.progress_reward_weights_by_metric,
            params={
                "metric_name": "flight_success_rate",
                "reward_weight_configs": {
                    "attitude_descent": {
                        "initial_weight": 1.0,
                        "target_weight": 5.0,
                        "metric_threshold": 0.7,
                        "metric_start": 0.3,
                    },
                },
                "min_steps_between_updates": 300,
                "smoothing_factor": 0.05,
            },
        ),
        
        # Phase 3: Focus on landing precision once flight is reliable
        "landing_phase": CurriculumTermCfg(
            func=curriculums.progress_reward_weights_by_metric,
            params={
                "metric_name": "full_jump_success_rate",
                "reward_weight_configs": {
                    "attitude_landing": {
                        "initial_weight": 0.1,
                        "target_weight": 2.0,
                        "metric_threshold": 0.9,
                        "metric_start": 0.5,
                    },
                    "feet_height": {
                        "initial_weight": 0.1,
                        "target_weight": 1.5,
                        "metric_threshold": 0.95,
                        "metric_start": 0.6,
                    },
                },
                "min_steps_between_updates": 500,
                "smoothing_factor": 0.02,
            },
        ),
    }

# Example usage in environment config:
"""
@configclass
class CurriculumCfg:
    # Choose one strategy
    reward_weights = precision_curriculum
    
    # Or use multiple phases
    # takeoff_rewards = multi_phase_curricula["takeoff_phase"]
    # flight_rewards = multi_phase_curricula["flight_phase"] 
    # landing_rewards = multi_phase_curricula["landing_phase"]
    
    # Keep command range curriculum
    command_ranges = CurriculumTermCfg(
        func=curriculums.progress_command_ranges,
        params={
            "num_curriculum_levels": 50,
            "success_rate_threshold": 0.9,
            "min_steps_between_updates": 150,
            "enable_regression": False,
        },
    )
""" 