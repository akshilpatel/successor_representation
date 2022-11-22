import numpy as np
import gym
from gym.spaces import Discrete
from typing import Union, Optional, Tuple, List
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

class GridWorld(gym.Env):
    def __init__(self, 
                layout: str,
                init_coords: Tuple, # coords
                terminal_coords: Tuple,
                reward_coords: dict,
                default_reward: Optional[Union[int, float]]=0.,
                stochasticity: Optional[Union[int, float]]=0., 
                seed: Optional[int] = None
                ):

        self.layout = layout
        self.layout_key = {'#': -1, '.': 0}

        self._action_to_direction = (np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])) # S, E, N, W
        self.grid, self.num_valid_states, self.coord_to_valid_state, self.valid_state_to_coord = self._process_layout(self.layout, self.layout_key)
        

        self._init_coords = tuple(s for s in init_coords if s not in terminal_coords)
        self._init_states = tuple([self._get_obs(s) for s in init_coords])
        self._reward_coords = reward_coords
        self._terminal_coords = terminal_coords
        
        self._terminal_states = tuple([self._get_obs(s) for s in self._terminal_coords])
        self._default_reward = default_reward

        if seed is None:
            seed = np.random.randint(0, 1000000)
        
        self.np_random = np.random.default_rng(seed)
        
        self.action_space = Discrete(4) # N, E, S, W        
        self.observation_space = Discrete(int(np.prod(self.grid.shape)))
        
        if stochasticity < 0 or stochasticity > 1:
            raise ValueError("Argument stochasticity must be a probability.")
        else:
            self.stochasticity = stochasticity        

    @staticmethod
    def _process_layout(layout: str, layout_key: dict) -> np.ndarray:
        """Generates default gridworld according to spec, and init, goal and terminal states.
        Args:
            layout (_type_): _description_
        """

        x_size = len(layout[0])
        y_size = len(layout)
        
        grid = np.zeros([y_size, x_size], dtype=int)
        
        for j, y in enumerate(layout):
            for i, x in enumerate(y):
                grid[j, i] = layout_key[x]

        num_valid_states = np.sum(grid==0)

        coord_to_valid_state_idx = {}
        
        counter = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 0:
                    coord_to_valid_state_idx[(i, j)] = counter
                    counter += 1

        valid_state_idx_to_coord = {v:k for k,v in coord_to_valid_state_idx.items()}

        return grid, num_valid_states, coord_to_valid_state_idx, valid_state_idx_to_coord
    
    def obs_to_valid_idx(self, obs):
        coord = self._obs_to_state(obs)
        idx = self.coord_to_valid_state[coord]
        return idx

    def _is_terminal(self) -> bool:
        # If the state is in the terminal states.
        obs = self._get_obs()
        return obs in self._terminal_states

    def _is_initial_state(self) -> bool:
        obs = self._get_obs()
        return obs in self._init_states
    
    def _get_info(self):
        info = {"agent_position": self._agent_pos}
        return info

    def is_action_ineffective(self, action) -> bool:
        action_d = self._action_to_direction[action]
        new_coords = self._agent_pos[0] + action_d[0], self._agent_pos[1] + action_d[1]
        
        # If the agent does not stay within vertical bounds.
        if new_coords[0] >= self.grid.shape[0] or new_coords[0] < 0:
            return True
        # If the agent does not stay within horizontal bounds.
        elif new_coords[1] >= self.grid.shape[1] or new_coords[1] < 0:
            return True
        # If the next_state is not a free space.
        else:
            return self.grid[new_coords] != 0
    
    
    def _get_reward(self):        
        if self._agent_pos in self._reward_coords:
            return self._reward_coords[self._agent_pos]
        else: 
            return self._default_reward

    def _get_obs(self, state=None):
        if state is None:
            state = self._agent_pos
        
        state = list(state)
        obs = np.ravel_multi_index(state, self.grid.shape).tolist()
        return obs
    
    def _obs_to_state(self, obs):
        return tuple(np.unravel_index(obs, self.grid.shape))

    
    def reset(self, seed:Optional[int]=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.action_space.seed = seed + 1
            self.observation_space.seed = seed + 2
        
        if len(self._init_coords) == 1:
            self._agent_pos = self._init_coords[0]
        else:
            self._agent_pos = tuple(self.np_random.choice(self._init_coords, 1).tolist()) # (i, j) position 

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def render(self, figsize=(10, 7)):
        render_grid = deepcopy(self.grid)
        fig = plt.figure(num = "env_render", figsize=figsize)
        
        ax = plt.gca()
        ax.clear()
        clear_output(wait=True)
        render_grid[self._agent_pos] = 1
        # Add terminal states

        # for coord in self._terminal_coords:
        #     plt.annotate("T", coord, va="center", ha="center", c="b")
       
        for coord, r_val in self._reward_coords.items():
            plt.annotate(str(r_val), coord[::-1], va="center", ha="center", c="gold" if coord in self._terminal_coords else "black")
        # Add numbers to rewarding states.
        
        ax.imshow(render_grid, cmap="magma")


        ax.grid(which = 'major', axis = 'both', linestyle = '-', color = 'k', linewidth = 2, zorder = 1)
        ax.set_xticks(np.arange(-0.5, render_grid.shape[1] , 1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, render_grid.shape[0], 1))
        ax.set_yticklabels([])
        plt.show()  
        
    # Finish and Test
    def step(self, action):
        """Step method.
        Args:
            action (_type_): _description_
        Returns:
            next_obs
            reward
            terminal
            truncated
            info
        """
        assert self.action_space.contains(action)
        # if it's stochastic - take random action
        if self.np_random.uniform(0, 1) < self.stochasticity:
            action = self.np_random.choice(range(self.action_space.n))

        # Ineffective actions make no change to state
        if not self.is_action_ineffective(action):
           action_d = self._action_to_direction[action]
           self._agent_pos = (self._agent_pos[0] + action_d[0], self._agent_pos[1] + action_d[1])

        next_obs = self._get_obs()
        reward = self._get_reward()
        terminal = self._is_terminal()
        info = self._get_info()
        return next_obs, reward, terminal, False, info


def generate_reward_profile_heatmap(transitions, env):
    # transitions is a state-action-reward etc.
    # 1) Extract next_state: rewards
    # 2) AVerage rewards according to next_state to get next_state : average reward
    # 3) Generate a grid array with values for the reward
    # 4) Overlay the grid onto a render.
    pass



class FourRooms(GridWorld):
    
    def __init__(
        self, 
        init_coords: Tuple, # coords
        terminal_coords: Tuple,
        reward_coords: dict,
        default_reward: float,
        stochasticity: Optional[Union[int, float]]=0., 
        seed: Optional[int] = None
        ):
        
        layout = np.loadtxt("env_layouts/four_rooms.txt", comments="//", dtype=str)
        super().__init__(layout, init_coords, terminal_coords, reward_coords, default_reward, stochasticity, seed)        


class DeltaVMaze(GridWorld):
    def __init__(
        self,
        init_coords: Tuple, # coords
        terminal_coords: Tuple,
        reward_coords: dict,
        default_reward: float,
        stochasticity: Optional[Union[int, float]]=0., 
        seed: Optional[int] = None
        ):

        layout = np.loadtxt("env_layouts/delta_v_maze.txt", comments="//", dtype=str)
        super().__init__(layout, init_coords, terminal_coords, reward_coords, default_reward, stochasticity, seed)

class ParrMaze(GridWorld):
    def __init__(
        self,
        init_coords: Tuple, # coords
        terminal_coords: Tuple,
        reward_coords: dict,
        default_reward: float,
        stochasticity: Optional[Union[int, float]]=0., 
        seed: Optional[int] = None
        ):
        layout = np.loadtxt("env_layouts/parr_maze.txt", comments="//", dtype=str)
        super().__init__(layout, init_coords, terminal_coords, reward_coords, default_reward, stochasticity, seed)


gym.envs.registration.register(
    id='FourRooms-v0',
    entry_point='gridworld:FourRooms',
)

gym.envs.registration.register(
    id='ParrMaze-v0',
    entry_point='gridworld:ParrMaze',
)
gym.envs.registration.register(
    id='DeltaVMaze-v0',
    entry_point='gridworld:DeltaVMaze',
)