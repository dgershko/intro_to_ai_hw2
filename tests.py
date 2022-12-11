import numpy as np
from pprint import pprint
import enum
import Gobblet_Gobblers_Env as gge



data = {'B1': (np.array([2, 0]), 'B'),
 'B2': (np.array([1, 1]), 'B'),
 'M1': (np.array([1, 2]), 'M'),
 'M2': (np.array([2, 1]), 'M'),
 'S1': (np.array([2, 0]), 'S'),
 'S2': (np.array([2, 2]), 'S')}

class size_to_value(enum.Enum):
    B = 3
    M = 2
    S = 1


# test_arr = np.zeros([3,3], dtype=int)
# for key, value in data.items():
#     (x, y) = value[0]
#     test_arr[y, x] = size_to_value[value[1]].value
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False
not_on_board = np.array([-1, -1])
# Returns a list of the current player's trios (each trio represents a row, a column or a diag in the board)
def get_unhidden_trios(state: gge.State, agent_id):
    player_pawns = [state.player1_pawns, state.player2_pawns]
    board = np.zeros([3,3], dtype=int)
    for key, value in player_pawns[agent_id].items():
        if is_hidden(state, agent_id, key) or np.array_equal(value[0], not_on_board):
            continue
        (x, y) = value[0]
        board[x, y] = size_to_value[value[1]].value
    trios = [np.zeros(3, dtype=int), np.zeros(3, dtype=int)]
    for i in range(3):
        trios.append(board[i,:])
        trios.append(board[:,i])
        trios[0][i] = board[i,i]
        trios[1][i] = board[i,2-i]
    return trios

env = gge.GridWorldEnv()
env.reset()
curr_state = env.get_state()
env.step(("B1", 0)) # P1: 0B
env.step(("S1", 4)) # P2: 4S
env.step(("M1", 4)) # P1: 0B & 4M
env.step(("B1", 4)) # P2: 4B
env.step(("S1", 2)) # P1: 0B & 2S
env.step(("M1", 2)) # P2: 4B & 2M
pprint(get_unhidden_trios(curr_state, 1))

# action which is a tuple of (pawn, location)
# chosen_step = agent_1(env.get_state(), PLAYER_ID, time_limit)
# action = chosen_step[0], int(chosen_step[1])

def eaten_board(state: gge.State, agent_id):
    player_pawns = [state.player1_pawns, state.player2_pawns]
    board = np.zeros([3,3], dtype=int)
    for key, value in player_pawns[agent_id].items():
        if not is_hidden(state, agent_id, key) or np.array_equal(value[0], not_on_board):
            continue
        (x, y) = value[0]
        board[x, y] = size_to_value[value[1]].value

def intermediate_heuristic(state: gge.State, agent_id):
    # Parameters and consts
    power_superparam = 1  # To make the difference between 
    eaten_weight_superparam = 1
    heuristic_value = 0
    position_value = 0
    eaten_value = 0
    
    # Sum trios values (by the formula stated in the dry section)
    for trio in get_unhidden_trios(state, agent_id):
        position_value += (np.count_nonzero(trio) ** power_superparam) * np.sum(trio)
    
    # 
    player_pawns = [state.player1_pawns, state.player2_pawns]
    for key, value in player_pawns[1-agent_id].items():
        if is_hidden(state, 1-agent_id, key):
            eaten_value += size_to_value[value[1]].value
    
    heuristic_value = position_value + eaten_weight_superparam * eaten_value
    return heuristic_value
    
    
