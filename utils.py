import operator
import numpy as np
import os

ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf


def get_directions():
    """Returns all the possible directions of a player in the game as a list of tuples.
    """
    return [(1, 0), (0, 1), (-1, 0), (0, -1)]


def tup_add(t1, t2):
    """
    returns the sum of two tuples as tuple.
    """
    return tuple(map(operator.add, t1, t2))


def get_board_from_csv(board_file_name):
    """Returns the board data that is saved as a csv file in 'boards' folder.
    The board data is a list that contains: 
        [0] size of board
        [1] blocked poses on board
        [2] starts poses of the players
    """
    board_path = os.path.join('boards', board_file_name)
    board = np.loadtxt(open(board_path, "rb"), delimiter=" ")
    
    # mirror board
    board = np.flipud(board)
    i, j = len(board), len(board[0])
    blocks = np.where(board == -1)
    blocks = [(blocks[0][i], blocks[1][i]) for i in range(len(blocks[0]))]
    start_player_1 = np.where(board == 1)
    start_player_2 = np.where(board == 2)
    
    if len(start_player_1[0]) != 1 or len(start_player_2[0]) != 1:
        raise Exception('The given board is not legal - too many start locations.')
    
    start_player_1 = (start_player_1[0][0], start_player_1[1][0])
    start_player_2 = (start_player_2[0][0], start_player_2[1][0])

    return [(i, j), blocks, [start_player_1, start_player_2]]


# Our Helper Classes/Functions
class State:
    def __init__(self, board, player_pos, rival_pos, turn_player, turn_count,
                 fruits, player_score, rival_score, fruits_taken=0):
        self.board = board
        self.player_pos = player_pos
        self.rival_pos = rival_pos
        self.turn_player = turn_player  # 1 = player, 2 = opponent
        self.turn_count = turn_count
        self.fruits = fruits
        self.player_score = player_score
        self.rival_score = rival_score
        self.fruits_taken = fruits_taken
        self.legal_moves = []

    def get_legal_moves(self, turn_player=1):
        if self.legal_moves:
            return self.legal_moves

        pos = self.player_pos if turn_player == 1 else self.rival_pos
        board = self.board

        legal_moves = []
        for d in get_directions():
            i = pos[0] + d[0]
            j = pos[1] + d[1]
            # check legal move
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and (board[i][j] not in [-1, 1, 2]):
                legal_moves.append(d)

        self.legal_moves = legal_moves
        return legal_moves

    def avail_steps_score(self):
        num_steps_available = len(self.get_legal_moves())
        if num_steps_available == 0:
            return -1
        else:
            return 4 - num_steps_available

    def man_dist_to_closest_fruit(self):
        man_dist = np.inf
        for fruit in self.fruits:
            man_dist_temp = abs(fruit[0] - self.player_pos[0]) + abs(fruit[1] - self.player_pos[1])
            man_dist = min(man_dist, man_dist_temp)
        if man_dist == np.inf:
            return 0
        return 1 / man_dist

    def dist_from_rival(self):
        man_dist = abs(self.rival_pos[0] - self.player_pos[0]) + abs(self.rival_pos[1] - self.player_pos[1])
        return 1 / man_dist

    def heuristic_weights(self, weights):
        return weights[0]*self.avail_steps_score() \
               + weights[1] * self.dist_from_rival() \
               + weights[2]*self.man_dist_to_closest_fruit() \
               + weights[3]*(self.player_score - self.rival_score) \
               + weights[4]*self.fruits_taken


def get_weights(simple=0.4, rival_dist=0.3, closest_fruit=0, score_diff=0.3, fruits_taken=0):
    return [simple, rival_dist, closest_fruit, score_diff, fruits_taken]
