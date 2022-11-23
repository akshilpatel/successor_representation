import numpy as np 
import pickle 
from collections import OrderedDict, Counter
from matplotlib import colors
from matplotlib import pyplot as plt
from copy import deepcopy
from successor_representation.agents import TabularCountsAgent
import pandas as pd
######################
# Runners
######################

def run_episodic(
    agent, 
    env, 
    num_episodes, 
    seed=32, 
    learn_sr=True, 
    save=False, 
    render=False,
    eval=False,
    episode_step_lim=10000,
    ):
    agent.set_seed(seed + 1)
    buffer = []
    epi_steps = []
    env.set_seed(seed)
    for i in range(num_episodes):
        
        obs, _ = env.reset()
        if isinstance(agent, TabularCountsAgent) and not eval:
            agent.update_counter(obs)

        for step in range(episode_step_lim):

            # Step the environment
            action = agent.choose_action(obs)
            next_obs, reward, done, *_ = env.step(action)
            if render: env.render()
                
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
            
            # Update
            if not eval:
                if isinstance(agent, TabularCountsAgent):
                    agent.update_counter(next_obs)
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
    render=False,
    eval=False,
    episode_step_lim=10000
    ):

    buffer = []
    epi_steps = []
    agent.set_seed(seed + 1)
    env.set_seed(seed)

    obs, _ = env.reset()
    if isinstance(agent, TabularCountsAgent) and not eval:
        agent.update_counter(obs)

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
        
        if render: env.render()
        
        buffer.append(transition)
        if not eval:
            if isinstance(agent, TabularCountsAgent):
                agent.update_counter(next_obs)
            agent.update(transition)
            if learn_sr:
                agent.update_sr(transition)


        if learn_sr:            
            agent.update_sr(transition)        
        
        # Switching between if the agent can continue from terminal state or reset.
        if done or (i + 1) % episode_step_lim == 0:
            if terminating:
                epi_steps.append(i+1)
                next_obs, _ = env.reset()
            
        obs = deepcopy(next_obs)
        
    if save:
        out_dir = "results/{}_env/{}_agent/{}_interactions/".format(env.name, agent.name, num_interactions)
        with open(out_dir + buffer, "wb") as f:
            pickle.dump(buffer, f)

    # Since the agent can go to a terminating state then leave and immediately return, this is meeaningless.
    if not terminating:
        epi_steps = []
    return buffer, agent, epi_steps

######################
# Maths
######################

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

class StandardScaling:
    def __init__(self):
        self.mean = 0.
        self.var = 0.
        self.n = 0
        self.s = 0
    
    def fit(self, x):
        x = np.asarray(x)
        self.n +=1
        if self.n <= 1:
            self.mean = x
            self.var = x ** 2
        else:
            diff = x - self.mean
            self.mean += diff / self.n
            self.s += diff * (x - self.mean)

            self.var = self.s / (self.n - 1)

    def fit_transform(self, x):
        self.fit(x)

        x_out = (x - self.mean) 

        if self.var != 0:
            x_out /= np.sqrt(self.var)

        return x_out

###########################################
# PLOTTING 
###########################################

def plot_heatmap(env, coord_to_heatmap, title, default_opacity=0., figsize=(12,8)):
    fig, ax = plt.subplots(figsize=figsize)
    render_colours  = OrderedDict(
                block_colour= "#202020", # black -1
                space_colour= "#fff7cc",  # off-white 0
                terminal_colour= "#ffd700", # 1 muted red
                agent_colour= "#507051", #  # 2 orange
            )
                        
    bounds = list(range(3))
    cmap = colors.ListedColormap(render_colours.values())
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    render_grid = deepcopy(env.grid)
    render_grid[env._agent_pos] = 2

    # Colour terminal states
    for coord in env._terminal_coords:
        render_grid[coord] = 1

    # Annotate init_state
    for coord in env._init_coords:
        ax.annotate("S", coord[::-1], va="center", ha="center", c="black", weight = "bold")

    # Annotate rewarded states
    for coord, r_val in env._reward_coords.items():
        if coord in env._terminal_coords:
            colour = "red"
        else:
            colour = "black"
        ax.annotate(str(r_val), coord[::-1], va="center", ha="center", c=colour) # gold

    # Annotate rewarded states
    for coord in env._terminal_coords:
        if coord not in env._reward_coords:
            ax.annotate("T", coord[::-1], va="center", ha="center", c="red") # gold

    ax.imshow(render_grid, cmap=cmap)


    ax.grid(which = 'major', axis = 'both', linestyle = '-', color = render_colours["block_colour"], linewidth = 2, zorder = 1)
    ax.set_xticks(np.arange(-0.5, render_grid.shape[1] , 1))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(-0.5, render_grid.shape[0], 1))
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    
    heatmap_grid = deepcopy(env.grid)
    opacity = np.full_like(heatmap_grid, default_opacity, dtype=np.float32)
    wall_coords = np.where(env.grid == -1)
    
    for coord in zip(*wall_coords):
        opacity[coord] = 0.

    for coord, val in coord_to_heatmap.items():            
        heatmap_grid[coord] = val
        opacity[coord] = 1.

    plt.title(title)
    heat_map = ax.imshow(heatmap_grid, cmap="viridis", alpha=opacity, zorder=1)
    fig.colorbar(heat_map)
    return fig, ax


    
def plot_sr_heatmap(sr, env, obs_coord, figsize=(12,8)):
    
    title="SR values from state {}".format(obs_coord)
    obs_valid = env.coord_to_valid_state[obs_coord]
    sr_vec = sr[obs_valid, :]
    sr_coord_to_heatmap = {env.valid_state_to_coord[i]: sr_vec[i] for i in range(sr_vec.shape[0])}

    # State from which to measure SR
    fig, ax = plot_heatmap(env, sr_coord_to_heatmap, title, 0, figsize)
    ax.annotate("X", obs_coord[::-1], va="center", ha="center", c="black", weight = "bold")
    
    
    plt.show()

    
    
def plot_state_visitation_heatmap(buffer, env, figsize=(12,8)):
    
    title= "State visitation for {} transitions".format(len(buffer))

    counter = Counter()
    
    for transition in buffer:
        counter[env._obs_to_state(transition["obs"])] += 1
    counter[env._obs_to_state(buffer[-1]["next_obs"])] += 1

    # State from which to measure SR
    fig, ax = plot_heatmap(env, counter,  title, 1., figsize)    
    
    plt.show()


def plot_reward_profile(buffer, env, reward_name, figsize=(12,8)):
        
    df = pd.DataFrame(buffer)
    df["next_coord"] = df["next_obs"].apply(env._obs_to_state)
    rewards_df = df[["next_coord", "reward"]].groupby("next_coord").apply("mean")["reward"]
    title="{} profile over {} transitions".format(reward_name, len(buffer))
    # State from which to measure SR
    fig, ax = plot_heatmap(env, rewards_df.to_dict(), title, 1, figsize)

    plt.show()