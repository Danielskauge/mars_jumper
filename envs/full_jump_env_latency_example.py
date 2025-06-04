# Example configuration showing how to use latency-aware observations
# This is a copy of the main config with latency examples enabled

from mars_jumper.envs.full_jump_env_cfg import *
from terms import observations

@configclass
class ObservationsCfgWithLatency:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # Standard observations (no latency)
        base_height = ObservationTermCfg(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_rotation_vector = ObservationTermCfg(func=observations.base_rotation_vector)
        has_taken_off = ObservationTermCfg(func=observations.has_taken_off)
        command_vec = ObservationTermCfg(func=observations.takeoff_height_length_cmd, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Latency-aware observations with different delays
        base_lin_vel_delayed = ObservationTermCfg(
            func=observations.base_lin_vel_latency,
            params={"latency_ms": 50.0},  # 50ms latency
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        
        base_ang_vel_delayed = ObservationTermCfg(
            func=observations.base_ang_vel_latency,
            params={"latency_ms": 30.0},  # 30ms latency
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        
        joint_pos_delayed = ObservationTermCfg(
            func=observations.joint_pos_rel_latency,
            params={"latency_ms": 25.0},  # 25ms latency
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        
        joint_vel_delayed = ObservationTermCfg(
            func=observations.joint_vel_rel_latency,
            params={"latency_ms": 40.0},  # 40ms latency
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        
        previous_actions_delayed = ObservationTermCfg(
            func=observations.last_action_latency,
            params={"latency_ms": 20.0}  # 20ms latency
        )

        def __post_init__(self):
            self.enable_corruption = False 
            self.concatenate_terms = True 

    policy: PolicyCfg = PolicyCfg()


@configclass
class FullJumpEnvCfgWithLatency(FullJumpEnvCfg):
    """Environment configuration with latency-aware observations."""
    
    # Override observations with latency-aware version
    observations: ObservationsCfgWithLatency = ObservationsCfgWithLatency()


# Usage example:
# To use this configuration instead of the standard one, import this class:
# from mars_jumper.envs.full_jump_env_latency_example import FullJumpEnvCfgWithLatency
# 
# Then use FullJumpEnvCfgWithLatency instead of FullJumpEnvCfg when creating your environment. 