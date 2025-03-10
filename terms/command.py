from typing import Sequence
from isaaclab.managers import CommandTerm, CommandTermCfg
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.utils import configclass
from dataclasses import MISSING
import logging
logger = logging.getLogger(__name__)
        
class TakeoffVelVecCommand(CommandTerm):
    """Command generator that generates a takeoff velocity vector command [num_envs, 2]. index 0 is pitch, index 1 is magnitude"""
    cfg: CommandTermCfg
    
    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2, device=self.device) #pitch, magnitude
        
    def _resample_command(self, env_ids: Sequence[int]):
        """ Actual resampling is handeled by reset_command in events.py, 
        as the command is needed in both the events terms for state initialization, 
        which are run before the command manager is reset """
        self._command[env_ids] = self._env._command_buffer[env_ids]
        
    
    @property
    def command(self) -> torch.Tensor:
        return self._command
    
    def _update_command(self):
        pass
    
    def _update_metrics(self):
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass 
    
    def _debug_vis_callback(self, event):
        pass
    

@configclass
class TakeoffVelVecCommandCfg(CommandTermCfg):
    """Configuration for the takeoff velocity vector command."""
    class_type: type = TakeoffVelVecCommand
    
    asset_name: str = MISSING
    
    @configclass
    class Ranges: 
        pitch_rad: tuple[float, float] = MISSING
        magnitude: tuple[float, float] = MISSING #meters per second
        
    ranges: Ranges = MISSING
    
    #TODO add vector visualizers
        