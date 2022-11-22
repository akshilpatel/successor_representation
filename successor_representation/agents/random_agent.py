import numpy as np

class SRRandomAgent:
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

    def update_sr(self, transition):
        obs = transition["valid_obs"]
        next_obs = transition["valid_next_obs"]
    
        kron_ohe = np.zeros(self.num_valid_obs)
        kron_ohe[obs] = 1
    
        deltas = self.sr_lr * (kron_ohe + self.sr_gamma * self.sr[next_obs] - self.sr[obs])
        self.sr[obs] += deltas