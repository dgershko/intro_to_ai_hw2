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

global calls
calls = 0

@functools.lru_cache(maxsize=256)
def smart_heuristic(state: gge.State, agent_id):
    # # Parameters and consts
    # # power_superparam = 1  # By increasing this we're making sure the agent favors combos (eg. 2 in a row)
    # # eaten_weight_superparam = 1  # controls the weighting of eating vs getting combos
    global calls
    calls += 1
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

# class Minimaxer():
#     def __init__(self, caller_agent):
#         self.agent_id = caller_agent
#         self.current_agent = caller_agent

#     # Get the state's value as viewed by a certain agent (min/max)
#     # The current agent doesn't affect final states
#     # The current agent does affect heuristic values
#     @functools.cache
#     def state_value(self, state: gge.State, depth):
#         match_state = gge.is_final_state(state)
#         if not match_state is None:
#             match_state = int(match_state) - 1
#             if match_state == self.agent_id:
#                 return math.inf
#             elif match_state == 1 - self.agent_id:
#                 return -math.inf
#             return 0
        
#         if depth == 0:
#             return smart_heuristic(state, self.agent_id)

#         if self.current_agent == self.agent_id:
#             return self.min_value(state, depth)
#         else:
#             return self.max_value(state, depth)

#     @functools.lru_cache(maxsize=128)
#     def max_value(self, state: gge.State, depth):
#         self.current_agent = 1 - self.current_agent
#         new_depth = depth - 1
#         value = max([self.state_value(neighbor[1], new_depth) for neighbor in state.get_neighbors()])
#         return value

#     @functools.lru_cache(maxsize=128)
#     def min_value(self, state: gge.State, depth):
#         self.current_agent = 1 - self.current_agent
#         new_depth = depth - 1
#         value = min([self.state_value(neighbor[1], new_depth) for neighbor in state.get_neighbors()]) 
#         return value

def rb_iteration(curr_state: gge.State, agent_id, depth):
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
        return max([rb_iteration(neighbor[1], agent_id, depth - 1) for neighbor in neighbor_list])
    else: # enemy turn
        return min([rb_iteration(neighbor[1], agent_id, depth - 1) for neighbor in neighbor_list])

        

def rb_iteration_wrapper(curr_state: gge.State, agent_id, depth, child_conn):
    global calls
    calls = 0
    begin = time.time()
    neighbor_list = curr_state.get_neighbors()
    best_action = neighbor_list[0][0]
    max_neighbor_value = -math.inf
    for neighbor in neighbor_list:
        move_value = rb_iteration(neighbor[1], agent_id, depth - 1)
        if move_value > max_neighbor_value:
            best_action = neighbor[0]
            max_neighbor_value = move_value
    print(f"depth: {depth}, elapsed time: {time.time() - begin}")
    print(f"calls: {calls}")
    child_conn.send(best_action)

def rb_heuristic_min_max(curr_state: gge.State, agent_id, time_limit):
    start_time = time.time()
    depth = 1
    parent_conn, child_conn = multiprocessing.Pipe() # ayy lmao
    while True:
        process = multiprocessing.Process(target=rb_iteration_wrapper, args=(curr_state, agent_id, depth, child_conn))
        process.start()
        remaining_time = time_limit - (time.time() - start_time)
        process.join(remaining_time * 0.9)
        if process.is_alive(): # if job is still alive after the remaining time ran out, its time to /die/
            print("timeout")
            process.kill()
            print(f"best action: {best_action}")
            return best_action
        best_action = parent_conn.recv() # update best action after iteration is done
        depth +=1 # prepare for next iteration with more depth
    

def alpha_beta_iteration(curr_state: gge.State, agent_id, depth, alpha, beta):
    match_state = gge.is_final_state(curr_state)
    if not match_state is None:
        match_state = int(match_state) - 1
        if match_state == agent_id:
            return math.inf
        elif match_state == 1 - agent_id:
            return -math.inf
        return 0
    
    if depth <= 0:
        return smart_heuristic(curr_state, agent_id)
    
    neighbor_list = curr_state.get_neighbors()

    if curr_state.turn == agent_id:
        curr_max = -math.inf
        for neighbor in neighbor_list:
            curr_max = max(curr_max, alpha_beta_iteration(neighbor[1], agent_id, depth - 1, alpha, beta))
            alpha = max(curr_max, alpha)
            if curr_max >= beta:
                return math.inf
        return curr_max
    else:
        curr_min = math.inf
        for neighbor in neighbor_list:
            curr_min = min(curr_min, alpha_beta_iteration(neighbor[1], agent_id, depth - 1, alpha, beta))
            beta = min(curr_min, beta)
            if curr_min <= alpha:
                return -math.inf
        return curr_min

def alpha_beta_iteration_wrapper(curr_state: gge.State, agent_id, depth, child_conn):
    begin = time.time()
    neighbor_list = curr_state.get_neighbors()
    best_action = neighbor_list[0][0]
    
    # Check each neighbor for its value
    curr_max = -math.inf
    alpha = -math.inf
    beta = math.inf
    if depth == 0:
        best_action = greedy_improved(curr_state, agent_id, 0)
    else:
        for neighbor in neighbor_list:
            # curr_max = max(curr_max, alpha_beta_iteration(neighbor[1], agent_id, depth - 1, alpha, beta))            
            ab_value = alpha_beta_iteration(neighbor[1], agent_id, depth - 1, alpha, beta)
            if ab_value >= curr_max:
                curr_max = ab_value
                best_action = neighbor[0]
            alpha = max(curr_max, alpha)
    print(f"alpha beta: depth: {depth}, elapsed time: {time.time() - begin}")
    child_conn.send(best_action)

    
def alpha_beta(curr_state, agent_id, time_limit):
    start_time = time.time()
    depth = 1
    best_action = None
    parent_conn, child_conn = multiprocessing.Pipe() # ayy lmao
    while True:
        process = multiprocessing.Process(target=alpha_beta_iteration_wrapper, args=(curr_state, agent_id, depth, child_conn))
        process.start()
        remaining_time = time_limit - (time.time() - start_time)
        process.join(remaining_time * 0.9)
        if process.is_alive(): # if job is still alive after the remaining time ran out, its time to /die/
            print("timeout")
            process.kill()
            print(f"best action: {best_action}")
            return best_action
        best_action = parent_conn.recv() # update best action after iteration is done
        depth +=1 # prepare for next iteration with more depth


def expectimax_iteration(curr_state: gge.State, agent_id, depth):
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
        return max([expectimax_iteration(neighbor[1], agent_id, depth - 1) for neighbor in neighbor_list])
    else: # probabilistic!
        total_probability = 0
        expected_value = 0
        for neighbor in neighbor_list:
            probability = 1
            if neighbor[0][0] in ["S1", "S2"]:
                probability = 2
            if dumb_heuristic2(neighbor[1], agent_id) > dumb_heuristic2(curr_state, agent_id):  # TODO - change it if piazza says so
                probability = 2
            total_probability += probability
            expectivalue =  expectimax_iteration(neighbor[1], agent_id, depth - 1)
            expected_value += probability * expectivalue
            # print(f">> Current value: {expectivalue}")

        return expected_value/total_probability
        

def expectimax_iteration_wrapper(curr_state: gge.State, agent_id, depth, child_conn):
    global calls
    calls = 0
    begin = time.time()
    neighbor_list = curr_state.get_neighbors()
    best_action = neighbor_list[0][0]
    
    # Check each neighbor for its value
    curr_max = -math.inf
    for neighbor in neighbor_list:
        expecti_value = expectimax_iteration(neighbor[1], agent_id, depth - 1)
        if expecti_value >= curr_max:
            curr_max = expecti_value
            best_action = neighbor[0]
    print(f"expectimax: depth: {depth}, elapsed time: {time.time() - begin}")
    print(f"calls: {calls}")
    child_conn.send(best_action)

def expectimax(curr_state, agent_id, time_limit):
    start_time = time.time()
    depth = 1
    best_action = None
    parent_conn, child_conn = multiprocessing.Pipe() # ayy lmao
    while True:
        process = multiprocessing.Process(target=expectimax_iteration_wrapper, args=(curr_state, agent_id, depth, child_conn))
        process.start()
        remaining_time = time_limit - (time.time() - start_time)
        process.join(remaining_time * 0.9)
        if process.is_alive(): # if job is still alive after the remaining time ran out, its time to /die/
            print("timeout")
            process.kill()
            print(f"best action: {best_action}")
            return best_action
        best_action = parent_conn.recv() # update best action after iteration is done
        depth +=1 # prepare for next iteration with more depth

# these is the BONUS - not mandatory    
class SuperAgent():
    move_to_index = {
            ('B1', 0): 0, ('B2', 0): 1, ('M1', 0): 2, ('M2', 0): 3, ('S1', 0): 4, ('S2', 0): 5,
            ('B1', 1): 6, ('B2', 1): 7, ('M1', 1): 8, ('M2', 1): 9, ('S1', 1): 10, ('S2', 1): 11,
            ('B1', 2): 12, ('B2', 2): 13, ('M1', 2): 14, ('M2', 2): 15, ('S1', 2): 16, ('S2', 2): 17,
            ('B1', 3): 18, ('B2', 3): 19, ('M1', 3): 20, ('M2', 3): 21, ('S1', 3): 22, ('S2', 3): 23,
            ('B1', 4): 24, ('B2', 4): 25, ('M1', 4): 26, ('M2', 4): 27, ('S1', 4): 28, ('S2', 4): 29,
            ('B1', 5): 30, ('B2', 5): 31, ('M1', 5): 32, ('M2', 5): 33, ('S1', 5): 34, ('S2', 5): 35,
            ('B1', 6): 36, ('B2', 6): 37, ('M1', 6): 38, ('M2', 6): 39, ('S1', 6): 40, ('S2', 6): 41,
            ('B1', 7): 42, ('B2', 7): 43, ('M1', 7): 44, ('M2', 7): 45, ('S1', 7): 46, ('S2', 7): 47,
            ('B1', 8): 48, ('B2', 8): 49, ('M1', 8): 50, ('M2', 8): 51, ('S1', 8): 52, ('S2', 8): 53}
    index_to_move = []

    def __init__(self, agent_id, heuristic: callable, initial_state: gge.State, time_limit):
        self.heuristic = heuristic
        self.agent_id = agent_id
        self.time_limit = time_limit
        self.initial_state = initial_state
        self.neighbor_list = initial_state.get_neighbors()
    
    def super_agent_iteration(self, curr_state: gge.State, depth, alpha, beta):
        if self.stop_event.is_set():
            sys.exit()
        match_state = gge.is_final_state(curr_state)
        if not match_state is None:
            match_state = int(match_state) - 1
            if match_state == self.agent_id:
                return math.inf
            elif match_state == 1 - self.agent_id:
                return -math.inf
            return 0
        
        if depth <= 0:
            return self.heuristic(curr_state, self.agent_id)
        
        neighbor_list = curr_state.get_neighbors()

        if curr_state.turn == self.agent_id:
            curr_max = -math.inf
            for neighbor in neighbor_list:
                curr_max = max(curr_max, self.super_agent_iteration(neighbor[1], depth - 1, alpha, beta))
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return math.inf
            return curr_max
        else:
            curr_min = math.inf
            for neighbor in neighbor_list:
                curr_min = min(curr_min, self.super_agent_iteration(neighbor[1], depth - 1, alpha, beta))
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return -math.inf
            return curr_min

    def super_agent_iteration_wrapper(self):
        depth = 1
        while not self.stop_event.is_set():
            iteration_start = time.time()
            iteration_best_action = None
            curr_max = -math.inf
            alpha = -math.inf
            beta = math.inf
            if depth == 1:
                for neighbour in self.neighbor_list:
                    action_value = self.heuristic(neighbour[1], self.agent_id)
                    if action_value >= curr_max:
                        curr_max = action_value
                        iteration_best_action = neighbour[0]
            else:
                for neighbor in self.neighbor_list:
                    action_value = self.super_agent_iteration(neighbor[1], depth - 1, alpha, beta)
                    if action_value >= curr_max:
                        curr_max = action_value
                        iteration_best_action = neighbor[0]
                    alpha = max(curr_max, alpha)
            self.best_action = iteration_best_action
            print(f"reached depth: {depth}, iteration took: {time.time() - iteration_start}")
            depth += 1
        sys.exit()
    
    def run_agent(self):
        self.stop_event = threading.Event()
        start_time = time.time()
        self.best_action = None
        worker_thread = threading.Thread(target=self.super_agent_iteration_wrapper)
        worker_thread.start()
        remaining_time = self.time_limit - (time.time() - start_time)
        worker_thread.join(remaining_time * 0.97)
        if worker_thread.is_alive():
            self.stop_event.set()
        print(self.best_action)
        return self.best_action


class SuperAgentFactory():
    def __init__(self, heuristic_func: callable):
        self.heuristic_func = heuristic_func

    def super_agent(self, curr_state, agent_id, time_limit):
        agent = SuperAgent(agent_id, self.heuristic_func, curr_state, time_limit)
        return agent.run_agent()
