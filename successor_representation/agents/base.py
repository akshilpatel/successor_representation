from abc import ABCMeta, abstractmethod
import numpy as np

class Agent(metaclass = ABCMeta):
    
    @abstractmethod
    def update(self, batch):
        raise NotImplementedError()
        
    @abstractmethod
    def choose_action(self, obs):
        raise NotImplementedError()

    def update_sr(self, transition):
        obs = transition["valid_obs"]
        next_obs = transition["valid_next_obs"]
    
        kron_ohe = np.zeros(self.num_valid_obs)
        kron_ohe[obs] = 1
    
        deltas = self.sr_lr * (kron_ohe + self.sr_gamma * self.sr[next_obs] - self.sr[obs])
        self.sr[obs] += deltas

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)