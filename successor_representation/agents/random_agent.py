import numpy as np
from successor_representation.agents.base import Agent

class SRRandomAgent(Agent):
    def __init__(self, num_valid_obs, sr_lr= 0.25, sr_gamma= 0.99, num_actions=4, seed=12):
        self.num_actions = num_actions
        self.num_valid_obs = num_valid_obs
        self.rng = np.random.default_rng(seed)
        self.sr_gamma = sr_gamma
        self.sr_lr = sr_lr
        self.sr = np.zeros((num_valid_obs, num_valid_obs), dtype=np.float32)
        self.name = "random"
    
    def choose_action(self, obs):
        action = self.rng.integers(0, self.num_actions)
        return action
        
    def update(self, *args):
        return {}