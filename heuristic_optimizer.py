import time
import math
import sys
import threading
import random
import itertools
import multiprocessing
import functools
from numpy import average
from copy import deepcopy
from pprint import pprint

import Gobblet_Gobblers_Env as gge
from submission import is_hidden

class SuperAgent():
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
        
        # neighbor_list = curr_state.get_neighbors()
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
            # print(f"heuristic {self.heuristic.__str__()} reached depth: {depth}, iteration took: {time.time() - iteration_start}")
            depth += 1
        sys.exit()
    
    def run_agent(self):
        self.stop_event = threading.Event()
        start_time = time.time()
        self.best_action = None
        worker_thread = threading.Thread(target=self.super_agent_iteration_wrapper)
        worker_thread.start()
        remaining_time = self.time_limit - (time.time() - start_time)
        worker_thread.join(remaining_time - 0.1)
        if worker_thread.is_alive():
            self.stop_event.set()
        # print(self.best_action)
        return self.best_action


class SuperAgentFactory():
    def __init__(self, heuristic_func: callable):
        self.heuristic_func = heuristic_func

    def super_agent(self, curr_state, agent_id, time_limit):
        agent = SuperAgent(agent_id, self.heuristic_func, curr_state, time_limit)
        return agent.run_agent()

def silent_game(agent_1, agent_2, time_limit):
    total_silence = True
    env = gge.GridWorldEnv('human')
    winner = None
    env.reset()
    start_time = 0
    end_time = 0
    steps_per_game = 0
    while winner is None:
        if env.s.turn == 0:
            start_time = time.time()
            chosen_step = agent_1(env.get_state(), 0, time_limit)
            end_time = time.time()
            if chosen_step is None:
                continue
            action = chosen_step[0], int(chosen_step[1])
            if (end_time - start_time) > time_limit:
                raise RuntimeError("Agent used too much time!")
            env.step(action)
            steps_per_game += 1
            if not total_silence: print("time for step was", end_time - start_time)
        else:
            start_time = time.time()
            chosen_step = agent_2(env.get_state(), 1, time_limit)
            end_time = time.time()
            if chosen_step is None:
                continue
            action = chosen_step[0], int(chosen_step[1])
            if (end_time - start_time) > time_limit:
                raise RuntimeError("Agent used too much time!")
            env.step(action)
            steps_per_game += 1
            if not total_silence: print("time for step was", end_time - start_time)

        winner = gge.is_final_state(env.s)
        if steps_per_game >= 300:
            winner = 0
    if winner == 0:
        if not total_silence: print("tie")
    else:
        if not total_silence: print("winner is:", winner)
    return int(winner) - 1, steps_per_game


class heuristic_creator():
    def __init__(self, size_to_value: dict, eat_mult, enemy_mult, combo_mult):
        self.size_to_value = size_to_value
        self.eat_mult = eat_mult
        self.enemy_mult = enemy_mult
        self.combo_mult = combo_mult

    def __str__(self):
        return f"{str(self.size_to_value)}, {str(self.eat_mult)}, {str(self.enemy_mult)}, {str(self.combo_mult)}"

    @functools.cache
    def get_positional_value(self, pawn_tuple): # pawn tuple: [(x,y,B)]
        # row 0, 1, 2, col 0, 1, 2, diag 0, 1
        trios = [[], [], [], [], [], [], [], []]
        for pawn in pawn_tuple:
            if pawn[0] == -1:
                continue
            value = self.size_to_value[pawn[2]]
            trios[pawn[0]].append(value)
            trios[pawn[1] + 3].append(value)
            if pawn[0] == pawn[1]:
                trios[6].append(value)
            if pawn[0] == (2 - pawn[1]):
                trios[7].append(value)

        return sum([sum(trio) * (len(trio) ** self.combo_mult) for trio in trios])

    def __call__(self, state: gge.State, agent_id):
        heuristic_value = 0
        position_value = 0
        eaten_value = 0

        pawns = [
            tuple((value[0][0], value[0][1], value[1]) for key, value in state.player1_pawns.items() if not is_hidden(state, 0, key)),
            tuple((value[0][0], value[0][1], value[1]) for key, value in state.player2_pawns.items() if not is_hidden(state, 1, key))
        ]
        position_value = self.get_positional_value(pawns[agent_id]) - self.enemy_mult * self.get_positional_value(pawns[1 - agent_id])

        # Sum values of enemy eaten (hidden) goblins
        player_pawns = [state.player1_pawns, state.player2_pawns]
        for key, value in player_pawns[1-agent_id].items():
            if is_hidden(state, 1-agent_id, key):
                eaten_value += self.size_to_value[value[1]]

        heuristic_value = position_value + self.eat_mult * eaten_value
        return heuristic_value

def generate_heuristics(initial_heuristic_params):
    heuristic_params_list = [initial_heuristic_params]
    for _ in range(4):
        new_heuristic_params = deepcopy(initial_heuristic_params)
        new_heuristic_params[0]['B'] = random.randint(1, 9)
        new_heuristic_params[0]['M'] = random.randint(1, new_heuristic_params[0]['B'])            
        new_heuristic_params[0]['S'] = random.randint(1, new_heuristic_params[0]['S'])            
        new_heuristic_params[1] = random.randint(0, 6)            
        new_heuristic_params[2] = random.randint(0, 6)            
        new_heuristic_params[3] = random.randint(0, 6)            
        heuristic_params_list.append(new_heuristic_params)
    return heuristic_params_list

def run_agents(agents: list):
    winner, steps = silent_game(agents[0][0], agents[1][0], run_agents.time_limit)
    return agents[winner][1], steps

def compare_heuristics(heuristics: list[callable]):
    agents = []
    for heuristic in heuristics:
        agents.append((SuperAgentFactory(heuristic).super_agent, heuristic.__str__())) # [(agent, heurustic_name), ]
    agent_combinations = itertools.permutations(agents, 2) # all possible 2-tuples of agents: [(agent_1, agent_2), (agent_2, agent_3),]
    with multiprocessing.Pool() as process_pool:
        result = process_pool.map(run_agents, agent_combinations)
        process_pool.close()
        process_pool.join()
    
    pprint(result, indent=4)
    keys = [heuristic.__str__() for heuristic in heuristics]
    result_dict = {key: [0, []] for key in keys}
    for line in result:
        result_dict[line[0]][0] += 1
        result_dict[line[0]][1].append(line[1])
    for key in result_dict:
        result_dict[key][1] = average(result_dict[key][1])
    print("===================================================")
    print("result dict:")
    pprint(result_dict, indent=4)
    print("===================================================")
    return result_dict


def optimize_heuristic(epochs, time_per_step, initial_heuristic_params = [{'B': 3, 'M': 2, 'S': 1}, 5, 3, 3]):
    run_agents.time_limit = time_per_step
    epoch_best_params = initial_heuristic_params
    winner_params = []

    for epoch in range(1, epochs + 1):
        try:
            heuristic_param_list = generate_heuristics(epoch_best_params)
            if epoch % 5 == 0: # sanity check
                heuristic_param_list.append(initial_heuristic_params)
            print(f"best params for epoch {epoch}: {epoch_best_params}")
            result_dict = compare_heuristics([heuristic_creator(*param) for param in heuristic_param_list])
            epoch_best_params = list(result_dict.keys())[0]
            for key in result_dict: # result_dict[params]
                if result_dict[key][0] > result_dict[epoch_best_params][0]:
                    epoch_best_params = key
                elif result_dict[key][0] == result_dict[epoch_best_params][0]:
                    if result_dict[key][1] < result_dict[epoch_best_params][1]:
                        epoch_best_params = key
            winner_params.append(list(eval(epoch_best_params)))
        except KeyboardInterrupt:
            break
    print()
    print("winner params:")
    pprint(winner_params)
    return winner_params

if __name__ == "__main__":
    optimize_heuristic(10, 20)