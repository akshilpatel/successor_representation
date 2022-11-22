import numpy as np 
import pickle 
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
            epi_steps.append(i)
            # Switching between if the agent can continue from terminal state or reset.
            if terminating: 
                obs, _ = env.reset()
        else:
            obs = next_obs
        
    if save:
        out_dir = "results/{}_env/{}_agent/{}_interactions/".format(env.name, agent.name, num_interactions)
        with open(out_dir + buffer, "wb") as f:
            pickle.dump(buffer, f)

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