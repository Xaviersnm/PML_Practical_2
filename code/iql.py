from collections import defaultdict
import random
from typing import List, DefaultDict

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class IQL:
    """
    Agent using the Independent Q-Learning algorithm
    """
    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        epsilon: float = 1.0,
        **kwargs,
    ):
        """
        Constructor of IQL

        Initializes variables for independent Q-learning agents

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)
        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr n_acts (List[int]): number of actions for each agent
        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents
        """
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        # access value of Q_i(o, a) with self.q_tables[i][str((o, a))] (str conversion for hashable obs)
        self.q_tables: List[DefaultDict] = [
            defaultdict(lambda: 0) for _ in range(self.num_agents)
        ]

    def act(self, obss) -> List[int]:
        """
        Implement the epsilon-greedy action selection here for stateless task

        **IMPLEMENT THIS FUNCTION**

        :param obss (List): list of observations for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []

        ### PUT YOUR CODE HERE ###
        
        for agent in range(self.num_agents):
            curr_obs = obss[agent]

            if np.random.random() < self.epsilon: # Exploration
                action = np.random.randint(self.n_acts[agent]) # Select random action

            else: # Exploitation
                agent_q_values = []
                for a in range(self.n_acts[agent]):
                    agent_q_values.append(self.q_tables[agent][str((curr_obs, a))]) # Add all q values to the list to choose the max

                action = np.argmax(agent_q_values)

            actions.append(action)

        return actions

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """
        Updates the Q-tables based on agents' experience

        **IMPLEMENT THIS FUNCTION**

        :param obss (List[np.ndarray]): list of observations for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray]): list of observations after taking the action for each agent
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (List[float]): updated Q-values for current actions of each agent
        """
        ### PUT YOUR CODE HERE ###

        for agent in range(self.num_agents):
            # Get all info from current agent
            curr_obs = obss[agent]
            curr_action = actions[agent]
            reward = rewards[agent]
            next_obs = n_obss[agent]

            curr_q = self.q_tables[agent][str((curr_obs,curr_action))]

            # Get future q values to calculate current q value
            if done:
                max_fut_q = 0.0 # If the episode is done there is no future reward
            
            else:
                fut_q_values = []
                for a in range(self.n_acts[agent]):
                    fut_q_values.append(self.q_tables[agent][str((next_obs, a))])

                max_fut_q = np.max(fut_q_values)

            # Compute new q-value
            new_q = curr_q + self.learning_rate * (reward + (self.gamma * max_fut_q) - curr_q)

            # Update value
            self.q_tables[agent][str((curr_obs, curr_action))] = new_q


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99
