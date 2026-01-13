from collections import defaultdict
import random
from typing import List, DefaultDict

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class CQL:
    """
    Agent using the Central Q-Learning algorithm
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
        Constructor of CQL

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

        # Single Joint Q-Table
        # Access value of Q(joint_obs, joint_actions) with self.q_table[str((joint_obs, joint_actions))]
        self.q_table = defaultdict(lambda: 0)

    def act(self, obss) -> List[int]:
        """
        Implement the epsilon-greedy action selection here for stateless task

        **IMPLEMENT THIS FUNCTION**

        :param obss (List): list of observations for each agent
        :return (List[int]): index of selected action for each agent
        """

        ### PUT YOUR CODE HERE ###
        
        if np.random.random() < self.epsilon: # Exploration
            joint_actions = []
            for agent in range(self.num_agents):
                joint_actions.append(np.random.randint(self.n_acts[agent]))
            
            return joint_actions

        else: # Exploitation
            all_joint_actions = [[]]
            for agent in range(self.num_agents):
                
                new_combinations = []

                for combination in all_joint_actions:
                    for action in range(self.n_acts[agent]):
                        new_combinations.append(combination + [action])

                all_joint_actions = new_combinations
            
            max_q_value = -float('inf') # Set first value to -inf (smallest possible)
            best_joint_action = all_joint_actions[0]

            for action_list in all_joint_actions:
                action_tuple = tuple(action_list)
                key = str((obss, action_tuple))
                q_val = self.q_table[key]
                if q_val > max_q_value:
                    max_q_value = q_val
                    best_joint_action = action_tuple
            
            return list(best_joint_action)


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
        # Get all info
        current_key = str((obss, tuple(actions)))
        current_q = self.q_table[current_key]

        # Get Global Reward
        total_reward = sum(rewards)

        # Get future q values to calculate current q value
        if done:
            max_fut_q = 0.0
        else:
            all_joint_actions = [[]]
            for agent in range(self.num_agents):
                new_combinations = []
                for combination in all_joint_actions:
                    for action in range(self.n_acts[agent]):
                        new_combinations.append(combination + [action])
                all_joint_actions = new_combinations

            max_fut_q = -float('inf')
            
            for action_list in all_joint_actions:
                action_tuple = tuple(action_list)
                future_key = str((n_obss, action_tuple))
                
                q_val = self.q_table[future_key]
                if q_val > max_fut_q:
                    max_fut_q = q_val
            
            # If the Q-table was empty for the next state, max_fut_q might still be -inf.
            if max_fut_q == -float('inf'):
                max_fut_q = 0.0

        # Compute new q-value
        new_q = current_q + self.learning_rate * (
            total_reward + (self.gamma * max_fut_q) - current_q
        )

        # Update value
        self.q_table[current_key] = new_q


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99
