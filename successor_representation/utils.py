import numpy as np 
import pickle 
from collections import OrderedDict
from matplotlib import colors
from matplotlib import pyplot as plt
from copy import deepcopy


def run_episodic(
    agent, 
    env, 
    num_episodes, 
    seed=32, 
    learn_sr=True, 
    save=False, 
    episode_step_lim=10000
    ):
    
    buffer = []
    epi_steps = []
    env.set_seed(seed)
    for i in range(num_episodes):
        
        obs, _ = env.reset()
        for step in range(episode_step_lim):
            action = agent.choose_action(obs)
            next_obs, reward, done, *_ = env.step(action)
            transition = {
                "obs" : obs,
                "action": action,
                "reward": reward,
                "done": done,
                "next_obs": next_obs,    
                "valid_obs": env.obs_to_valid_idx(obs),
                "valid_next_obs": env.obs_to_valid_idx(next_obs)
            }
            buffer.append(transition)
            agent.update(transition)
            if learn_sr:
                agent.update_sr(transition)
            

            if done or (step + 1) % episode_step_lim == 0 :
                epi_steps.append(step)    
                obs, _ = env.reset()
                break
            else:
                obs = next_obs
    
    if save:
        out_dir = "results/{}/{}_{}_epi_".format(env.name, agent.name, num_episodes)
        with open(out_dir + "buffer.pickle", "wb+") as f:
            pickle.dump(buffer, f)
    return buffer, agent, epi_steps


def run_non_episodic(
    agent, 
    env, 
    num_interactions, 
    seed=32, 
    save=False, 
    learn_sr=True, 
    terminating=True, 
    episode_step_lim=10000
    ):

    buffer = []
    epi_steps = []
    env.set_seed(seed)

    obs, _ = env.reset()
    
    for i in range(num_interactions):
        action = agent.choose_action(obs)
        next_obs, reward, done, *_ = env.step(action)
        transition = {
            "obs" : obs,
            "action": action,
            "reward": reward,
            "done": done,
            "next_obs": next_obs,
            "valid_obs": env.obs_to_valid_idx(obs),
            "valid_next_obs": env.obs_to_valid_idx(next_obs),
        }
        buffer.append(transition)
        agent.update(transition)
        if learn_sr:            
            agent.update_sr(transition)        
        
        if done or (i + 1) % episode_step_lim == 0:
            epi_steps.append(i+1)
            # Switching between if the agent can continue from terminal state or reset.
            if terminating: 
                
                obs, _ = env.reset()
        else:
            obs = deepcopy(next_obs)
        
    if save:
        out_dir = "results/{}_env/{}_agent/{}_interactions/".format(env.name, agent.name, num_interactions)
        with open(out_dir + buffer, "wb") as f:
            pickle.dump(buffer, f)

    # Since the agent can go to a terminating state then leave and immediately return, this is meeaningless.
    if not terminating:
        epi_steps = []
    return buffer, agent, epi_steps


def compute_sr_vec(lr, gamma, buffer, num_obs):
    psi = np.zeros((num_obs, num_obs), dtype=np.float32)
    
    for transition in buffer:
        obs = transition["valid_obs"]
        next_obs = transition["valid_next_obs"]
        
        kron_ohe = np.zeros(num_obs)
        kron_ohe[obs] = 1.
        
        deltas = lr * (kron_ohe + gamma * psi[next_obs] - psi[obs])
        psi[obs] += deltas
    return psi


def render_sr(env, sr, obs_coord, title, figsize=(10, 7)):
        render_colours  = OrderedDict(
            block_colour= "#101010", # black -1
            space_colour= "#fff7cc",  # off-white 0
        )
                
        # Creating a colour map?
        bounds = list(range(3))
        cmap = colors.ListedColormap(render_colours.values())
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        render_grid = deepcopy(env.grid)
        # Creating the figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        obs_valid = env.coord_to_valid_state[obs_coord]
        
        # Rendering the grid with specified colours as the base layer
        ax.imshow(render_grid, cmap=cmap, norm=norm, zorder=0)

        # Adding lines in a grid on top
        ax.grid(which = 'major', axis = 'both', linestyle = '-', color = 'k', linewidth = 2, zorder = 1)
        
        ax.set_xticks(np.arange(-0.5, render_grid.shape[1] , 1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, render_grid.shape[0], 1))
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)

        # State from which to measure SR
        ax.annotate("X", obs_coord[::-1], va="center", ha="center", c="black", weight = "bold")
        
        # Annotate init_state
        if obs_coord not in env._init_coords:
            for coord in env._init_coords:
                ax.annotate("I", coord[::-1], va="center", ha="center", c="black", weight = "bold")

        # Annotate rewarded states
        for coord in env._terminal_coords:
            ax.annotate("T", coord[::-1], va="center", weight="bold", ha="center", c="black") # gold

        sr_grid = deepcopy(render_grid)
        sr_vec = sr[obs_valid, :]
        opacity = np.zeros_like(sr_grid, dtype=np.float32)


        for i, val in enumerate(sr_vec):
            tmp_coord = env.valid_state_to_coord[i]
            sr_grid[tmp_coord] = val
            opacity[tmp_coord] = 1.


        plt.title(title)
        sr_heat_map = ax.imshow(sr_grid, cmap="viridis", alpha=opacity, zorder=1)
        fig.colorbar(sr_heat_map)

        plt.show()