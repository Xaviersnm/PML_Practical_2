import copy
import random
import gymnasium as gym
import numpy as np
import lbforaging
from PIL import Image

from iql import IQL
from utils import visualise_evaluation_returns

CONFIG = {
    "seed": 0,
    "gamma": 0.99,
    "total_eps": 25000,      
    "ep_length": 50,        
    "eval_freq": 1000,    
    "lr": 0.1,              
    "init_epsilon": 1.0,
    "eval_epsilon": 0.05,
}

def preprocess_obss(obss):
    """
    Converts raw numpy observations (floats) into hashable tuples of integers to ensure
    the Q-table keys are stable (e.g., "1.0" vs "1.0001").
    """
    processed = []
    for obs in obss:
        processed.append(tuple(np.round(obs).astype(int)))
    return processed

def record_video(env_name, agent, config, filename="agent_gameplay.gif"):
    """
    Generates the video (.gif) of the episode.
    I got help from Gemini to pull this one off
    """
    print(f"Recording video to {filename}...")
    env = gym.make(env_name)
    video_agent = copy.deepcopy(agent)
    video_agent.epsilon = config["eval_epsilon"]

    frames = []
    obss, _ = env.reset()
    obss = preprocess_obss(obss)

    try:
        first_frame = env.unwrapped.render(mode="rgb_array")
        if first_frame is not None:
            frames.append(Image.fromarray(first_frame))
    except Exception:
        pass
    
    done = False
    step = 0
    
    while not done and step < config["ep_length"]:
        actions = video_agent.act(obss)
        n_obss, _, done, _, _ = env.step(actions)
        
        obss = preprocess_obss(n_obss)
        
        try:
            frame = env.unwrapped.render(mode="rgb_array")
            if frame is not None:
                frames.append(Image.fromarray(frame))
        except Exception:
             pass
        step += 1
    
    if len(frames) > 0:
        frames[0].save(
            filename, save_all=True, append_images=frames[1:],
            optimize=False, duration=200, loop=0
        )
        print(f"Done! Saved {len(frames)} frames.")
    
    try:
        env.close()
    except Exception:
        pass

def iql_eval(env, config, q_tables, eval_episodes=100, output=True):
    eval_agents = IQL(
        num_agents=env.unwrapped.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["eval_epsilon"],
    )
    eval_agents.q_tables = q_tables

    episodic_returns = []
    for _ in range(eval_episodes):
        obss, _ = env.reset()
        obss = preprocess_obss(obss)
        
        episodic_return = np.zeros(env.unwrapped.n_agents)
        done = False
        step = 0

        while not done and step < config["ep_length"]:
            actions = eval_agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(actions)
            
            obss = preprocess_obss(n_obss)
            
            episodic_return += rewards
            step += 1

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns, axis=0)
    std_return = np.std(episodic_returns, axis=0)

    if output:
        print(f"EVAL (Ep {_}): Agent 1: {mean_return[0]:.2f}, Agent 2: {mean_return[1]:.2f}")
    return mean_return, std_return

def train(env_name, config, output=True):
    print(f"\n--- Training IQL on {env_name} ---")
    env = gym.make(env_name)

    agents = IQL(
        num_agents=env.unwrapped.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]

    evaluation_return_means = []
    evaluation_return_stds = []

    for eps_num in range(config["total_eps"]):
        obss, _ = env.reset()
        obss = preprocess_obss(obss)
        
        episodic_return = np.zeros(env.unwrapped.n_agents)
        done = False
        step = 0

        while not done and step < config["ep_length"]:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            
            n_obss, rewards, done, _, _ = env.step(acts)
            
            n_obss = preprocess_obss(n_obss)
            
            agents.learn(obss, acts, rewards, n_obss, done)

            step_counter += 1
            episodic_return += rewards
            obss = n_obss
            step += 1

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            print(f"Episode {eps_num}/{config['total_eps']}")
            mean_return, std_return = iql_eval(
                env, config, agents.q_tables, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)

    return evaluation_return_means, evaluation_return_stds, agents

if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Task 1: Competitive
    print("\n>>> RUNNING TASK 1A: COMPETITIVE IQL")
    
    means_comp, stds_comp, trained_agents_comp = train("Foraging-5x5-2p-1f-v3", CONFIG)
    
    record_video(
        "Foraging-5x5-2p-1f-v3", 
        trained_agents_comp, 
        CONFIG, 
        filename="iql_competitive.gif"
    )
    
    visualise_evaluation_returns(means_comp, stds_comp)
    
    # Task 1: Cooperative 
    print("\n>>> RUNNING TASK 1B: COOPERATIVE IQL")
    means_coop, stds_coop, trained_agents_coop = train("Foraging-5x5-2p-1f-coop-v3", CONFIG)
    
    record_video(
        "Foraging-5x5-2p-1f-coop-v3", 
        trained_agents_coop, 
        CONFIG, 
        filename="iql_cooperative.gif"
    )
    
    visualise_evaluation_returns(means_coop, stds_coop)