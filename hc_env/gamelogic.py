from typing import Tuple
import numpy as np
import constants


class HoneyCombGame:
    """
    Encaspulates much of the game logic required for the HoneyComb environment.
    """
    def __init__(self, num_agents: int = 10, num_moves: int = 15):
        self.gamefield = None
        self.agents = num_agents
        self.moves = num_moves
        self.turns_count = None
        self.remaining_moves = None
        self.current_agent = None
        self.agent_list = []
        self.episode = 0
        self.storage = None

    def reset_game(self):
        """
        Reset the honeycomb playing field.
        """
        # Reset agent data
        self.agent_positions = {i: constants.START_POSITION for i in range(self.agents)}
        self.storage = {i: [constants.START_POSITION] for i in range(self.agents)}
        self.remaining_moves = {i: self.moves for i in range(self.agents)}
        self.turns_count = 1
        self.current_agent = 0
        self.episode += 1

    def sNorm(self, i: int, j: int) -> int:
        """
        Compute the distance from the origin on the playing field.
        :param i: coordinate 1
        :param j: coordinate 2
        :return: distance from origin 
        """
        if i < 0:
            i = -i
            j = -j
        if j < 0:
            return i - j
        elif i < j:
            return j
        else:
            return i

    def submit_move_for_agent(self, agent: int, move: Tuple[int, int]) -> bool:
        """
        Submit a more for agent i to validate it and if valid, execute it.
        :param agent: Index of agent
        :param move: displacement caused by move
        :return: bool indicating validity of move
        """
        d_r, d_c = move
        r_cur, c_cur = self.agent_positions[agent]
        r_temp, c_temp = r_cur + d_r, c_cur + d_c

        if (
            self.sNorm(r_temp, c_temp) <= 5
            or (r_temp, c_temp) in constants.PAYOFF_LOCATIONS
        ):
            self.agent_positions[agent] = (r_temp, c_temp)
            self.storage[agent].append((r_temp, c_temp))
            self.remaining_moves[agent] -= 1
            return True
        else:
            self.storage[agent].append(self.agent_positions[agent])
            self.remaining_moves[agent] -= 1
            return False

    def movable_agent(self):
        """
        Determines all agents with remaining moves.
        :return: list of moveable agents
        """
        a_list = []
        for a in range(self.agents):
            if self.remaining_moves[a] > 0:
                a_list.append(a)

        return a_list

    def get_agent_pos(self, agent: int) -> Tuple[int, int]:
        """
        Get the position of agent i.
        :param agent: Index of agent
        :return: current position of agent i
        """
        return self.agent_positions[agent]

    def check_all_arrived(self) -> bool:
        """
        Check whether all agents have arrived on a goal field.
        :return: bool indicating whether all have arrived
        """
        for agent in range(self.agents):
            if self.agent_positions[agent] not in constants.PAYOFF_LOCATIONS:
                return False
        return True
