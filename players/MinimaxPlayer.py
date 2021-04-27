"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer

import utils
import numpy as np
import time
from SearchAlgos import MiniMax


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None
        self.fruits_turns = 0
        self.player_pos = None
        self.rival_pos = None
        self.turn_count = 0
        self.fruits = []
        self.solver = MiniMax(self.utility, self.succ, self.perform_move)

        # time related variables
        self.game_start = 0         # estimate of time at which the game start
        self.elapsed_time = 0       # total elapsed game time
        self.time_left = None       # function returning time left in the current turn
        self.TIME_THRESHOLD = 1     # how many seconds to finish turn before time limit is reached

        # heuristic weights
        self.weights = utils.get_weights(simple=0.3, rival_dist=0.2, closest_fruit=0.1,
                                         score_diff=0.2, fruits_taken=0.2)

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = board.copy()
        self.fruits_turns = 2 * min(len(self.board[0]), len(self.board)) + 1

        coordinates = np.where(board == 1)
        self.player_pos = tuple(xy[0] for xy in coordinates)

        coordinates = np.where(board == 2)
        self.rival_pos = tuple(xy[0] for xy in coordinates)

        coordinates = np.where(board > 2)
        for x, y in zip(coordinates[0], coordinates[1]):
            fruit = (x, y)
            self.fruits.append(fruit)

        # if self.penalty_score < 3*min_fruit or self.penalty_score < max_fruit:
        #    self.weights = utils.get_weights(simple=0.2, rival_dist=0, closest_fruit=0.3,
        #                                     score_diff=0.5)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifying the Player's movement, chosen from self.directions
        """
        start = time.time()
        if self.turn_count == 0:
            self.game_start = start
            self.TIME_THRESHOLD = 0.05 * time_limit

        if self.turn_count == 1:
            self.TIME_THRESHOLD = 0.05 * time_limit

        # create state object for the minimax algorithm
        start_state = utils.State(self.board.copy(), self.player_pos, self.rival_pos, 1, self.turn_count,
                                   self.fruits.copy(), players_score[0], players_score[1])

        self.time_left = lambda: time_limit - (time.time() - start)
        legal_moves = start_state.get_legal_moves()     # we assume there's at least one legal move
        direction = legal_moves[0]                          # initialize arbitrarily

        try:
            if len(legal_moves) == 1:
                raise TimeoutError

            depth = 1
            while True:
                minimax, direction = self.solver.search(start_state, depth, True)
                depth += 1
        finally:
            self.perform_move(direction)                        # update the board
            self.elapsed_time = (time.time() - self.game_start)   # update leftover game time
            return direction

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        if self.turn_count == 0:
            self.game_start = time.time()

        self.board[self.rival_pos] = -1     # mark rival's current location as a grey area
        self.board[pos] = 2                 # mark the rival's current location
        self.rival_pos = pos                # save rival's new position
        self.turn_count += 1                # update turn count

        self.elapsed_time = (time.time() - self.game_start)   # update leftover game time

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        # if all fruits disappeared, remove them from board
        if not fruits_on_board_dict.keys():
            # change heuristic weights after fruits disappear
            # self.weights = utils.get_weights(simple=0.6, rival_dist=0.2, closest_fruit=0,
            #                                 score_diff=0.2)

            for fruit in self.fruits:
                if self.board[fruit] not in [1, -1, 2, 0]:
                    self.board[fruit] = 0

        self.fruits = list(fruits_on_board_dict.keys())

    # helper functions in class #
    def check_time_over(self):
        if self.time_left() < self.TIME_THRESHOLD or self.game_time - self.elapsed_time < self.TIME_THRESHOLD:
            raise TimeoutError  # reached time limit for this turn

    # helper functions for MiniMax algorithm #
    def utility(self, state, is_goal):
        if is_goal:
            ret_val = state.player_score - state.rival_score

            # check if is tie
            other_player_avail_moves = len(state.get_legal_moves(3 - state.turn_player))
            if other_player_avail_moves == 0:
                return ret_val

            if state.turn_player == 1:
                ret_val -= self.penalty_score
            else:
                ret_val += self.penalty_score

            return ret_val
        else:
            # heuristic utility
            return state.heuristic_weights(self.weights)

    def succ(self, state):
        """"
            state = (board, player_pos, rival_pos, turn_player, turn_count,
                     fruit_positions, player_score, rival_score)
        """
        self.check_time_over()

        pos = state.player_pos if state.turn_player == 1 else state.rival_pos
        new_turn_player = 3 - state.turn_player

        successors = []
        legal_moves = state.get_legal_moves(state.turn_player)
        for d in legal_moves:
            new_player_score = state.player_score
            new_rival_score = state.rival_score

            new_pos = utils.tup_add(pos, d)
            new_board = state.board.copy()
            new_fruits = state.fruits.copy()

            fruit_score = new_board[new_pos]
            new_fruits_taken = state.fruits_taken
            if fruit_score > 2:  # if is fruit
                new_fruits.remove(new_pos)
                new_fruits_taken += 1

            new_board[pos] = -1
            new_board[new_pos] = state.turn_player

            if state.turn_player == 1:
                new_player_pos, new_rival_pos = new_pos, state.rival_pos
                new_player_score += fruit_score
            else:
                new_player_pos, new_rival_pos = state.player_pos, new_pos
                new_rival_score += fruit_score

            if state.turn_count + 1 == self.fruits_turns:
                # fruits disappeared
                for fruit_pos in new_fruits:
                    new_board[fruit_pos] = 0
                new_fruits = []

            new_state = utils.State(new_board, new_player_pos, new_rival_pos, new_turn_player,
                                    state.turn_count + 1, new_fruits, new_player_score, new_rival_score,
                                    new_fruits_taken)

            successors.append(new_state)

        return successors

    def perform_move(self, direction):
        self.turn_count += 1  # update turn count
        self.board[self.player_pos] = -1
        self.player_pos = utils.tup_add(self.player_pos, direction)

        self.board[self.player_pos] = 1
