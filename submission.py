import random
import math
import time
import multiprocessing
import functools

import numpy as np

import Gobblet_Gobblers_Env as gge

not_on_board = np.array([-1, -1])

import enum

class Superparams():
    combo_weight = 3
    eating_weight = 1
    enemy_heuristic_weight = 1

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

@functools.lru_cache(maxsize=256)
def smart_heuristic(state: gge.State, agent_id):
    # # Parameters and consts
    # # power_superparam = 1  # By increasing this we're making sure the agent favors combos (eg. 2 in a row)
    # # eaten_weight_superparam = 1  # controls the weighting of eating vs getting combos
    heuristic_value = 0
    position_value = 0
    eaten_value = 0

    #TODO: make these three calculations parrallel?
    position_value = get_positional_value(state, agent_id) - get_positional_value(state, 1 - agent_id)
    
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
    max_heuristic = -math.inf
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


#==========================================#
# ---------------RB MINIMAX--------------- #
#==========================================#

class Minimaxer():
    def __init__(self, caller_agent):
        self.agent_id = caller_agent
        self.current_agent = caller_agent

    # Get the state's value as viewed by a certain agent (min/max)
    # The current agent doesn't affect final states
    # The current agent does affect heuristic values
    @functools.cache
    def state_value(self, state: gge.State, depth):
        match_state = gge.is_final_state(state)
        if match_state is not None:
            match_state = int(match_state) - 1
            if match_state == self.agent_id:
                return math.inf
            elif match_state == 1 - self.agent_id:
                return -math.inf
            return 0
        
        if depth == 0:
            return smart_heuristic(state, self.agent_id)

        if self.current_agent == self.agent_id:
            return self.min_value(state, depth)
        else:
            return self.max_value(state, depth)

    @functools.lru_cache(maxsize=128)
    def max_value(self, state: gge.State, depth):
        self.current_agent = 1 - self.current_agent
        new_depth = depth - 1
        value = max([self.state_value(neighbor[1], new_depth) for neighbor in state.get_neighbors()])
        return value

    @functools.lru_cache(maxsize=128)
    def min_value(self, state: gge.State, depth):
        self.current_agent = 1 - self.current_agent
        new_depth = depth - 1
        value = min([self.state_value(neighbor[1], new_depth) for neighbor in state.get_neighbors()]) 
        return value


def rb_heuristic_min_max(curr_state: gge.State, agent_id, time_limit):
    def rb_iteration(neighbor_list, agent_id, depth, child_conn):
        begin = time.time()
        minimaxer = Minimaxer(agent_id)
        best_action = None
        max_neighbor_value = -math.inf
        for neighbor in neighbor_list:
            move_value = minimaxer.state_value(neighbor[1], depth)
            if move_value > max_neighbor_value:
                best_action = neighbor[0]
                max_neighbor_value = move_value
        print(f"depth: {depth}, elapsed time: {time.time() - begin}")
        child_conn.send(best_action)

    #TODO: missing default move in case of instant timeout?
    start_time = time.time()
    neighbor_list = curr_state.get_neighbors()
    depth = 0
    best_action = None
    parent_conn, child_conn = multiprocessing.Pipe() # ayy lmao
    while True:
        minimax_iteration_process = multiprocessing.Process(target=rb_iteration, args=(neighbor_list, agent_id, depth, child_conn))
        minimax_iteration_process.start()
        remaining_time = time_limit - (time.time() - start_time)
        minimax_iteration_process.join(remaining_time - 0.05)
        if minimax_iteration_process.is_alive(): # if job is still alive after the remaining time ran out, its time to /die/
            print("timeout")
            minimax_iteration_process.kill()
            print(f"best action: {best_action}")
            return best_action
        best_action = parent_conn.recv() # update best action after iteration is done
        depth +=1 # prepare for next iteration with more depth
    

def alpha_beta(curr_state, agent_id, time_limit):
    raise NotImplementedError()


def expectimax(curr_state, agent_id, time_limit):
    raise NotImplementedError()

# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    raise NotImplementedError()
