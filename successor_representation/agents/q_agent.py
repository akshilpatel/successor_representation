import numpy as np
from collections import defaultdict

class SRQLearningAgent:
    # Assume discrete/Multidiscrete action and obs space.
    def __init__(self, num_valid_obs, num_actions=4, sr_lr=0.2, sr_gamma=0.999, gamma=0.99, lr=0.5, seed=42, min_epsilon=1e-4, max_epsilon=1., epsilon_decay=0.99):
        self.num_actions = num_actions
        self.num_valid_obs = num_valid_obs
        self.gamma = gamma
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self._qfa = defaultdict(lambda: [0. for _ in range(self.num_actions)])
        self.min_epsilon       = min_epsilon
        self.max_epsilon       = max_epsilon
        self.eps_decay_rate    = epsilon_decay
        self.epsilon = max_epsilon
        self.sr_gamma = sr_gamma
        self.sr_lr = sr_lr
        self.sr = np.zeros((num_valid_obs, num_valid_obs), dtype=np.float32)
        self.name = "tabular_q"

    def update(self, batch):
        obs      = batch["obs"]
        action   = batch["action"]
        rewards  = batch["reward"]
        next_obs = batch["next_obs"]
        dones    = batch["done"]
        
        next_obs_val = max(self._qfa[next_obs])                                           
        q_target  = rewards + self.gamma * (1 - dones) * next_obs_val                       

        self._qfa[obs][action]  = self._qfa[obs][action] * (1-self.lr) + q_target * self.lr

        return self._qfa[obs][action]

        
    def choose_action(self, obs):       
        # Greedy action
        self.decay_epsilon()
        if self.rng.random() > self.epsilon:
            return self.get_greedy_action(obs)
        # Random action
        else:    
            return self.rng.integers(self.num_actions)

    def get_greedy_action(self, obs):
        q_vals     = np.array(self._qfa[obs])
        max_q      = max(q_vals)
        action = self.rng.choice(np.where(q_vals == max_q)[0]) 
        return action

    def decay_epsilon(self):
        self.epsilon *= self.eps_decay_rate
        self.epsilon = np.clip(self.epsilon, self.min_epsilon, self.max_epsilon)

        return self.epsilon
    
    def update_sr(self, transition):
        obs = transition["valid_obs"]
        next_obs = transition["valid_next_obs"]
    
        kron_ohe = np.zeros(self.num_valid_obs)
        kron_ohe[obs] = 1.
    
        deltas = self.sr_lr * (kron_ohe + self.sr_gamma * self.sr[next_obs] - self.sr[obs])
        self.sr[obs] += deltas