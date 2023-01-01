import random
import math
import time
import multiprocessing
import functools
import threading
import sys

import numpy as np

import Gobblet_Gobblers_Env as gge

not_on_board = np.array([-1, -1])

import enum

class Superparams():
    combo_weight = 3
    eating_weight = 5
    enemy_heuristic_weight = 3

    class size_to_value(enum.Enum):
        B = 3
        M = 2
        S = 1


# agent_id is which player I am, 0 - for the first player , 1 - if second player
def dumb_heuristic1(state, agent_id):
    is_final = gge.is_final_state(state)
    # this means it is not a final state
    if is_final is None:
        return 0
    # this means it's a tie
    if is_final is 0:
        return -1
    # now convert to our numbers the win
    winner = int(is_final) - 1
    # now winner is 0 if first player won and 1 if second player won
    # and remember that agent_id is 0 if we are first player  and 1 if we are second player won
    if winner == agent_id:
        # if we won
        return 1
    else:
        # if other player won
        return -1


# checks if a pawn is under another pawn
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False


# checks if a pawn is over another pawn
def is_hiding(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == -1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == -1:
            return True
    return False


# count the numbers of pawns that i have that aren't hidden
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1

    return sum_pawns

def run_with_timer(func, time_limit, curr_state, agent_id):
    stop_event = threading.Event()
    start_time = time.time()
    best_action = [None]
    worker_thread = threading.Thread(target=func, args=[stop_event, curr_state, agent_id, best_action])
    worker_thread.start()
    remaining_time = time_limit - (time.time() - start_time)
    worker_thread.join(remaining_time - 0.2)
    if worker_thread.is_alive():
        stop_event.set()
    return best_action[0]

def get_positional_value(state: gge.State, agent_id):
    player_pawns = [state.player1_pawns, state.player2_pawns]
    board = np.zeros([3,3], dtype=int)
    for key, value in player_pawns[agent_id].items():
        if is_hidden(state, agent_id, key) or np.array_equal(value[0], not_on_board):
            continue
        (x, y) = value[0]
        board[x, y] = Superparams().size_to_value[value[1]].value
    trios = [board.diagonal(), np.fliplr(board).diagonal()]
    for i in range(3):
        trios.extend([board[i,:],board[:,i]])
    
    value = 0
    for trio in trios:
        value += (np.count_nonzero(trio) ** Superparams().combo_weight) * np.sum(trio)
    return value


def smart_heuristic(state: gge.State, agent_id):
    # # Parameters and consts
    # # power_superparam = 1  # By increasing this we're making sure the agent favors combos (eg. 2 in a row)
    # # eaten_weight_superparam = 1  # controls the weighting of eating vs getting combos
    heuristic_value = 0
    position_value = 0
    eaten_value = 0

    position_value = get_positional_value(state, agent_id)
    
    # Sum values of enemy eaten (hidden) goblins
    player_pawns = [state.player1_pawns, state.player2_pawns]
    for key, value in player_pawns[1-agent_id].items():
        if is_hidden(state, 1-agent_id, key):
            eaten_value += Superparams().size_to_value[value[1]].value
    
    heuristic_value = position_value + Superparams().eating_weight * eaten_value
    return heuristic_value


# IMPLEMENTED FOR YOU - NO NEED TO CHANGE
def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


# agent_id is which agent you are - first player or second player
def random_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    rnd = random.randint(0, neighbor_list.__len__() - 1)
    return neighbor_list[rnd][0]


# TODO - instead of action to return check how to raise not_implemented
def greedy(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = dumb_heuristic2(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]

# TODO - add your code here
def greedy_improved(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = -10000
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


def rb_iteration(stop_event: threading.Event, curr_state: gge.State, agent_id, depth):
    if stop_event.is_set():
        sys.exit()
    match_state = gge.is_final_state(curr_state)
    if not match_state is None:
        match_state = int(match_state) - 1
        if match_state == agent_id:
            return 1000000
        elif match_state == 1 - agent_id:
            return -1000000
        return 0
    
    if depth <= 0:
        return smart_heuristic(curr_state, agent_id)
    
    neighbor_list = curr_state.get_neighbors()
    
    if curr_state.turn == agent_id:
        return max([rb_iteration(stop_event, neighbor[1], agent_id, depth - 1) for neighbor in neighbor_list])
    else: # enemy turn
        return min([rb_iteration(stop_event, neighbor[1], agent_id, depth - 1) for neighbor in neighbor_list])


def rb_iteration_wrapper(stop_event: threading.Event, curr_state: gge.State, agent_id, best_action):
    neighbor_list = curr_state.get_neighbors()
    depth = 1
    while not stop_event.is_set():
        iteration_best_action = None
        curr_max = -math.inf
        if depth == 1:
            for neighbour in neighbor_list:
                action_value = smart_heuristic(neighbour[1], agent_id)
                if action_value >= curr_max:
                    curr_max = action_value
                    iteration_best_action = neighbour[0]
        else:
            for neighbor in neighbor_list:
                action_value = rb_iteration(stop_event, neighbor[1], agent_id, depth - 1)
                if action_value >= curr_max:
                    curr_max = action_value
                    iteration_best_action = neighbor[0]
        best_action[0] = iteration_best_action
        depth += 1
    sys.exit()

def rb_heuristic_min_max(curr_state: gge.State, agent_id, time_limit):
    return run_with_timer(rb_iteration_wrapper, time_limit, curr_state, agent_id)
    

def alpha_beta_iteration(stop_event: threading.Event, curr_state: gge.State, agent_id, depth, alpha, beta):
    if stop_event.is_set():
        sys.exit()
    match_state = gge.is_final_state(curr_state)
    if not match_state is None:
        match_state = int(match_state) - 1
        if match_state == agent_id:
            return 10000
        elif match_state == 1 - agent_id:
            return -10000
        return 0
    
    if depth <= 0:
        return smart_heuristic(curr_state, agent_id)
    
    neighbor_list = curr_state.get_neighbors()

    if curr_state.turn == agent_id:
        curr_max = -10000
        for neighbor in neighbor_list:
            curr_max = max(curr_max, alpha_beta_iteration(stop_event, neighbor[1], agent_id, depth - 1, alpha, beta))
            alpha = max(curr_max, alpha)
            if curr_max >= beta:
                return 10000
        return curr_max
    else:
        curr_min = 10000
        for neighbor in neighbor_list:
            curr_min = min(curr_min, alpha_beta_iteration(stop_event, neighbor[1], agent_id, depth - 1, alpha, beta))
            beta = min(curr_min, beta)
            if curr_min <= alpha:
                return -10000
        return curr_min

def alpha_beta_iteration_wrapper(stop_event: threading.Event, curr_state: gge.State, agent_id, best_action):
    neighbor_list = curr_state.get_neighbors()
    depth = 1
    while not stop_event.is_set():
        iteration_best_action = None
        curr_max = -math.inf
        alpha = -math.inf
        beta = math.inf
        if depth == 1:
            for neighbour in neighbor_list:
                action_value = smart_heuristic(neighbour[1], agent_id)
                if action_value >= curr_max:
                    curr_max = action_value
                    iteration_best_action = neighbour[0]
        else:
            for neighbor in neighbor_list:
                action_value = alpha_beta_iteration(stop_event, neighbor[1], agent_id, depth - 1, alpha, beta)
                if action_value >= curr_max:
                    curr_max = action_value
                    iteration_best_action = neighbor[0]
                alpha = max(curr_max, alpha)
        best_action[0] = iteration_best_action
        depth += 1
    sys.exit()

    
def alpha_beta(curr_state, agent_id, time_limit):
    return run_with_timer(alpha_beta_iteration_wrapper, time_limit, curr_state, agent_id)


def expectimax_iteration(stop_event: threading.Event, curr_state: gge.State, agent_id, depth):
    if stop_event.is_set():
        sys.exit()
    match_state = gge.is_final_state(curr_state)
    if not match_state is None:
        match_state = int(match_state) - 1
        if match_state == agent_id:
            return 1000000
        elif match_state == 1 - agent_id:
            return -1000000
        return 0
    
    if depth <= 0:
        return smart_heuristic(curr_state, agent_id)
    
    neighbor_list = curr_state.get_neighbors()
    
    if curr_state.turn == agent_id:
        return max([expectimax_iteration(stop_event, neighbor[1], agent_id, depth - 1) for neighbor in neighbor_list])
    else: # probabilistic!
        total_probability = 0
        expected_value = 0
        for neighbor in neighbor_list:
            probability = 1
            if neighbor[0][0] in ["S1", "S2"]:
                probability = 2
            if is_hiding(neighbor[1], neighbor[0][0], agent_id):
                probability = 2
            total_probability += probability
            expectivalue =  expectimax_iteration(stop_event, neighbor[1], agent_id, depth - 1)
            expected_value += probability * expectivalue

        return expected_value/total_probability
        

def expectimax_iteration_wrapper(stop_event: threading.Event, curr_state: gge.State, agent_id, best_action):
    neighbor_list = curr_state.get_neighbors()
    depth = 1
    while not stop_event.is_set():
        iteration_best_action = None
        curr_max = -math.inf
        if depth == 1:
            for neighbour in neighbor_list:
                expecti_value = smart_heuristic(neighbour[1], agent_id)
                if expecti_value >= curr_max:
                    curr_max = expecti_value
                    iteration_best_action = neighbour[0]
        else:
            for neighbor in neighbor_list:
                expecti_value = expectimax_iteration(stop_event, neighbor[1], agent_id, depth - 1)
                if expecti_value >= curr_max:
                    curr_max = expecti_value
                    iteration_best_action = neighbor[0]
        best_action[0] = iteration_best_action
        depth += 1
    sys.exit()

def expectimax(curr_state, agent_id, time_limit):
    return run_with_timer(expectimax_iteration_wrapper, time_limit, curr_state, agent_id)


# these is the BONUS - not mandatory 
# selected hyperparameters:
# size_to_val = {'B': 7, 'M': 6, 'S': 5}
# eat = 4
# enemy = 0
# combo = 2   
@functools.cache
def super_get_positional_value(pawn_tuple): # pawn tuple: [(x,y,B)]
    size_to_value = {
        'B': 7,
        'M': 6,
        'S': 5
    }
    # row 0, 1, 2, col 0, 1, 2, diag 0, 1
    trios = [[], [], [], [], [], [], [], []]
    for pawn in pawn_tuple:
        if pawn[0] == -1:
            continue
        value = size_to_value[pawn[2]]
        trios[pawn[0]].append(value)
        trios[pawn[1] + 3].append(value)
        if pawn[0] == pawn[1]:
            trios[6].append(value)
        if pawn[0] == (2 - pawn[1]):
            trios[7].append(value)

    return sum([sum(trio) * (len(trio) ** 2) for trio in trios])

def super_heuristic(state: gge.State, agent_id):
    size_to_value = {
        'B': 7,
        'M': 6,
        'S': 5
    }
    position_value = 0
    eaten_value = 0

    pawns = [
        tuple((value[0][0], value[0][1], value[1]) for key, value in state.player1_pawns.items() if not is_hidden(state, 0, key)),
        tuple((value[0][0], value[0][1], value[1]) for key, value in state.player2_pawns.items() if not is_hidden(state, 1, key))
    ]
    position_value = super_get_positional_value(pawns[agent_id])

    # Sum values of enemy eaten (hidden) goblins
    player_pawns = [state.player1_pawns, state.player2_pawns]
    for key, value in player_pawns[1-agent_id].items():
        if is_hidden(state, 1-agent_id, key):
            eaten_value += size_to_value[value[1]]

    heuristic_value = position_value + 4 * eaten_value
    return heuristic_value

def super_agent_iteration(stop_event, curr_state: gge.State, agent_id, depth, alpha, beta):
    if stop_event.is_set():
        sys.exit()
    match_state = gge.is_final_state(curr_state)
    if not match_state is None:
        match_state = int(match_state) - 1
        if match_state == agent_id:
            return 10000
        elif match_state == 1 - agent_id:
            return -10000
        return 0
    
    if depth <= 0:
        return super_heuristic(curr_state, agent_id)
    
    neighbor_list = curr_state.get_neighbors()

    if curr_state.turn == agent_id:
        curr_max = -10000
        for neighbor in neighbor_list:
            curr_max = max(curr_max, super_agent_iteration(stop_event, neighbor[1], agent_id, depth - 1, alpha, beta))
            alpha = max(curr_max, alpha)
            if curr_max >= beta:
                return 10000
        return curr_max
    else:
        curr_min = 10000
        for neighbor in neighbor_list:
            curr_min = min(curr_min, super_agent_iteration(stop_event, neighbor[1], agent_id, depth - 1, alpha, beta))
            beta = min(curr_min, beta)
            if curr_min <= alpha:
                return -10000
        return curr_min

def super_agent_iteration_wrapper(stop_event: threading.Event, curr_state: gge.State, agent_id, best_action): # Note: this is only run by threads
    neighbor_list = curr_state.get_neighbors()
    depth = 1
    while not stop_event.is_set():
        iteration_best_action = None
        curr_max = -math.inf
        alpha = -math.inf
        beta = math.inf
        if depth == 1:
            for neighbour in neighbor_list:
                action_value = super_heuristic(neighbour[1], agent_id)
                if action_value >= curr_max:
                    curr_max = action_value
                    iteration_best_action = neighbour[0]
        else:
            for neighbor in neighbor_list:
                action_value = super_agent_iteration(stop_event, neighbor[1], agent_id, depth - 1, alpha, beta)
                if action_value >= curr_max:
                    curr_max = action_value
                    iteration_best_action = neighbor[0]
                alpha = max(curr_max, alpha)
        best_action[0] = iteration_best_action
        depth += 1
    sys.exit()

def super_agent(curr_state, agent_id, time_limit):
    stop_event = threading.Event()
    start_time = time.time()
    best_action = [None]
    worker_thread = threading.Thread(target=super_agent_iteration_wrapper, args=[stop_event, curr_state, agent_id, best_action])
    worker_thread.start()
    remaining_time = time_limit - (time.time() - start_time)
    worker_thread.join(remaining_time - 0.1)
    if worker_thread.is_alive():
        stop_event.set()
    return best_action[0]
    
