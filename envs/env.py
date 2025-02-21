from isaaclab.envs import ManagerBasedRLEnv

class MarsJumperEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)