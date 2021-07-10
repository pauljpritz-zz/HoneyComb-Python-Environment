import constants
import random
import numpy as np
import gym
from gym.spaces import Box
from pprint import PrettyPrinter
import copy
import pandas as pd
from gamelogic import HoneyCombGame
from collections import defaultdict
from typing import Tuple


class HoneyCombEnv(gym.Env):
    """
    The HoneyComb Game:
    The parameters of the game are set in the constants.py file.
    Several agents are positioned on a hexagonal playing field, where there
    are 7 actions available to them, i.e., moving to an adjacent field (6 actions)
    and a void action (1 action) that allows them to stay on the current field.
    The agents receive a reward for terminating the game on one of 6 reward fields
    located at the edges of the playing field.
    NUM_INFORMED agents (biased agents) are aware of a special payoff field, that increases their reward
    by a factor of 2.
    The agent's state is returned as a numpy vector, containing the positions of all
    agents as well as the distance to the reward fields. The biased agents are additionally aware of the 
    special payoff field.
    Actions are submitted as a numpy array of integers, where the integer at index i coresponds to agent
    i's action.
    """

    # NOTE: The make_states, step and compute_rewards functions have been adjusted to fit their code

    def __init__(
        self,
        max_turns: int = 15,
        alternative_states: int = True
    ):

        self.use_alternative_states = alternative_states
        self.agents = constants.NUM_UNINFORMED + constants.NUM_INFORMED
        self.action_space = self.agents * [gym.spaces.Discrete(7)]
        obs_dim = self.agents * 2 + len(constants.PAYOFF_LOCATIONS) * 2 + 2 + 1
        self.observation_space = self.agents * [Box(-np.inf, np.inf, shape=(obs_dim,))]
        self.num_informed = constants.NUM_INFORMED
        self.max_turns = max_turns
        self.game = HoneyCombGame(self.agents, max_turns)
        self.done = False

        # Call reset here to initialize gamefield
        self.reset()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, str]:
        """
        Implementation of the agent-environment interaction.
        Performs one time step of environment dynamics.
        :param actions: an action provided by the agent (one-hot index)
        :return: state, rewards, done, info
        """

        int_actions = [np.where(r == 1)[0][0] for r in np.vstack(actions)]
        for agent in range(self.agents):
            action = int_actions[agent]
            valid_move = self.game.submit_move_for_agent(
                agent, constants.ACTIONS[action]
            )

        self.turns_count += 1
        done = np.array([False] * self.agents).reshape(1, -1)

        if self.game.check_all_arrived() or self.turns_count >= self.max_turns:
            if self.game.check_all_arrived():
                print("reached goals after %i " % self.turns_count)
            done = np.array([True] * self.agents).reshape(1, -1)

        rewards = self.compute_reward(done)

        if self.use_alternative_states:
            states = self.make_alternative_states()
        else:
            states = self.make_states()

        return states, rewards, done, {"Not Implemented": ""}

    # At the end of an episode call reset_game
    def reset(self) -> np.ndarray:
        """
        At the end of an episode call reset to initialize a new playing field.
        :return: current state
        """
        # Initialize a new gamefield
        self.game.reset_game()
        self.turns_count = 0

        self.payoff_fields = constants.PAYOFF_LOCATIONS
        self.special_payoff_fields = random.sample(self.payoff_fields, 1)

        return self.make_alternative_states()

    def compute_reward(self, done: np.ndarray) -> np.ndarray:
        """
        Compute the rewards for each agent.
        :done: Array indicating the done status for each agent
        :return: rewards
        """
        rewards = [0.0 for i in range(self.agents)]
        rewarded_fields = defaultdict(float)
        if done[0][0]:
            # Get rewards for all rew fields with agents on
            for agent in range(self.agents):
                pos = self.game.get_agent_pos(agent)
                if pos in self.payoff_fields:
                    rewarded_fields[pos] += 1.0

            for agent in range(self.agents):
                pos = self.game.get_agent_pos(agent)
                rewards[agent] += rewarded_fields.get(pos, 0.0)
                if agent < self.num_informed and pos in self.special_payoff_fields:
                    rewards[agent] *= 2

        rewards = np.array(rewards).reshape(1, -1)
        return rewards

    def make_states(self) -> np.ndarray:
        """
        Reflects the current state of the environment, as seen by each agent.
        :return: array with states for all agents
        """
        state = []
        for agent in range(self.agents):
            r, c = self.game.get_agent_pos(agent)
            state.append(r)
            state.append(c)
        for field in constants.PAYOFF_LOCATIONS:
            r, c = field
            state.append(r)
            state.append(c)
        state.append(self.turns_count)
        for field in self.special_payoff_fields:
            r, c = field
            state.append(r)
            state.append(c)

        informed_state = np.array(state)
        uninformed_state = copy.deepcopy(informed_state)
        uninformed_state[-2:] = 0

        states = [
            informed_state if i < self.num_informed else uninformed_state
            for i in range(self.agents)
        ]
        states = np.stack(states, axis=0)

        return states

    def make_alternative_states(self) -> np.ndarray:
        """
        Reflects the current state of the environment, as seen by each agent.
        In this case, positions are normalised and distances to reward fields are 
        reported rather than their position on the playing field.
        :return: array with states for all agents
        """
        states = []
        for agent in range(self.agents):
            agent_state = []

            # Own distance
            r, c = self.game.get_agent_pos(agent)
            agent_state.append(r / 6)
            agent_state.append(c / 6)

            # Distances to others
            distances_r = [
                (r - pos[0]) / 12
                for key, pos in self.game.agent_positions.items()
                if key != agent
            ]
            distances_c = [
                (c - pos[1]) / 12
                for key, pos in self.game.agent_positions.items()
                if key != agent
            ]
            agent_state += distances_r
            agent_state += distances_c

            # Goal distances
            distances_goal_r = [(r - pos[0]) / 12 for pos in self.payoff_fields]
            distances_goal_c = [(c - pos[1]) / 12 for pos in self.payoff_fields]
            agent_state += distances_goal_r
            agent_state += distances_goal_c

            if agent < self.num_informed:
                agent_state.append((r - self.special_payoff_fields[0][0]) / 12)
                agent_state.append((c - self.special_payoff_fields[0][1]) / 12)
            else:
                agent_state += [0, 0]
            agent_state.append(self.max_turns - self.turns_count)
            states.append(np.array(agent_state))

        states = np.stack(states, axis=0)
        return states

    def render(self, **kwargs):
        """
        Prints gamefield as matrix.
        :return: A numpy array containing the gamefield.
        """
        return print(self.game.agent_positions)


if __name__ == "__main__":
    env = HoneyCombEnv()
    for i in range(50):
        actions = [random.randint(0, 6) for i in range(10)]
        one_hot_actions = [np.array([0,0,0,0,0,0,0]) for i in actions]
        for i in range(len(actions)):
            one_hot_actions[i][actions[i]] = 1
        one_hot_actions = np.array(one_hot_actions)
            
        print(actions)
        print(one_hot_actions)
        env.render()
        state, rew, done, info = env.step(one_hot_actions)
        print(state[0])
        print(state[8])
        print(env.game.agent_positions)
        print("----------------")
