from successor_representation.agents.q_agent import SRQLearningAgent
from collections import Counter
import numpy as np
# from successor_representation.utils import StandardScaling

# Use an exploratory policy for learning the SR and for getting good information for seeing what is reachable.
class TabularCountsAgent(SRQLearningAgent):
    def __init__(self, num_valid_obs, num_actions=4, sr_lr=0.2, sr_gamma=0.999, gamma=0.99, lr=0.5, seed=42, min_epsilon=1e-4, max_epsilon=1., epsilon_decay=0.99):
        super().__init__(num_valid_obs, num_actions=4, sr_lr=0.2, sr_gamma=0.999, gamma=0.99, lr=0.5, seed=42, min_epsilon=1e-4, max_epsilon=1., epsilon_decay=0.99)
        self.name = "tabular_counts"
        self.counter = Counter()
        self.scaler = StandardScaling()

    def update_counter(self, next_obs):
        self.counter[next_obs] += 1

    def update(self, batch):
        obs      = batch["obs"]
        action   = batch["action"]
        reward   = batch["reward"]
        next_obs = batch["next_obs"]
        dones    = batch["done"]
        
        count_reward = 1./ np.sqrt(self.counter[next_obs])
        
        self.scaler.fit_transform(count_reward)

        next_obs_val = max(self._qfa[next_obs])                                           
        q_target  = count_reward + self.gamma * (1 - dones) * next_obs_val                       

        self._qfa[obs][action]  = self._qfa[obs][action] * (1-self.lr) + q_target * self.lr

        return self._qfa[obs][action]






