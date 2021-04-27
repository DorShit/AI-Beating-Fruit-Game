"""Search Algos: MiniMax, AlphaBeta
"""
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
#TODO: you can import more modules, if needed
import numpy as np

class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        children = self.succ(state)

        is_goal = (children == [])
        if is_goal or depth == 0:
            return self.utility(state, is_goal), None

        if maximizing_player:
            cur_max = -np.inf
            direction = None
            max_pos = None
            for child in children:
                val, unused = self.search(child, depth-1, not maximizing_player)
                if val > cur_max:
                    cur_max = val
                    max_pos = child.player_pos
            if max_pos is not None:
                direction = (max_pos[0] - state.player_pos[0], max_pos[1] - state.player_pos[1])
            return cur_max, direction
        else:
            cur_min = np.inf
            for child in children:
                val, unused = self.search(child, depth-1, not maximizing_player)
                cur_min = min(cur_min, val)
            return cur_min, None


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        children = self.succ(state)

        is_goal = (children == [])
        if is_goal or depth == 0:
            return self.utility(state, is_goal), None

        if maximizing_player:
            cur_max = -np.inf
            direction = None
            for child in children:
                val, unused = self.search(child, depth - 1, not maximizing_player, alpha, beta)
                if val > cur_max:
                    cur_max = val
                    max_pos = child.player_pos
                    direction = (max_pos[0] - state.player_pos[0], max_pos[1] - state.player_pos[1])
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return np.inf, None
            return cur_max, direction
        else:
            cur_min = np.inf
            for child in children:
                val, unused = self.search(child, depth - 1, not maximizing_player, alpha, beta)
                cur_min = min(cur_min, val)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return -np.inf, None
            return cur_min, None
