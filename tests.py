import numpy as np
from pprint import pprint
import enum
import Gobblet_Gobblers_Env as gge
import time
import math
import signal
import multiprocessing
import asyncio
import threading
import os
import ctypes
import sys
import functools
from _thread import interrupt_main

from submission import alpha_beta, SuperAgentFactory, is_hidden, human_agent
from submission import get_positional_value as default_pos_val
from utils import game

not_on_board = np.array([-1, -1])

trio_value_dict = {(): 0,
                   (1,): 1,
                   (1, 1): 16,
                   (1, 2): 24,
                   (1, 3): 32,
                   (2,): 2,
                   (2, 1): 24,
                   (2, 2): 32,
                   (2, 3): 40,
                   (3,): 3,
                   (3, 1): 32,
                   (3, 2): 40,
                   (3, 3): 48}


class size_to_value(enum.Enum):
    B = 3
    M = 2
    S = 1


@functools.cache
def trio_value(a=0, b=0, c=0):
    trio = [a, b, c]
    return len(trio) * sum(trio)


arr = np.array([1, 2, 0])

'''
size_cmp gets two possible sizes of pawns meaning B or M or S
and returns
 1 if size1 > size2
-1 if size2 > size1
0 if size1 == size2 
'''


def get_neighbors_super(state: gge.State):
    def size_cmp(size1, size2):
        if size1 == size2:
            return 0
        if size1 == "B":
            return 1
        if size1 == "S":
            return -1
        if size2 == "S":
            return 1
        else:
            return -1

    def find_curr_location(curr_state, pawn, player):
        if player == 0:
            for pawn_key, pawn_value in curr_state.player1_pawns.items():
                if pawn_key == pawn:
                    return pawn_value[0]
        else:
            for pawn_key, pawn_value in curr_state.player2_pawns.items():
                if pawn_key == pawn:
                    return pawn_value[0]

    action_to_direction = {
        0: np.array([0, 0]),
        1: np.array([0, 1]),
        2: np.array([0, 2]),
        3: np.array([1, 0]),
        4: np.array([1, 1]),
        5: np.array([1, 2]),
        6: np.array([2, 0]),
        7: np.array([2, 1]),
        8: np.array([2, 2]),
    }

    def is_legal_step(action, curr_state: gge.State):
        pawn_list = {
            "agent1_big1": [curr_state.player1_pawns["B1"][0], "B"],
            "agent1_big2": [curr_state.player1_pawns["B2"][0], "B"],
            "agent1_medium1": [curr_state.player1_pawns["M1"][0], "M"],
            "agent1_medium2": [curr_state.player1_pawns["M2"][0], "M"],
            "agent1_small1": [curr_state.player1_pawns["S1"][0], "S"],
            "agent1_small2": [curr_state.player1_pawns["S2"][0], "S"],
            "agent2_big1": [curr_state.player2_pawns["B1"][0], "B"],
            "agent2_big2": [curr_state.player2_pawns["B2"][0], "B"],
            "agent2_medium1": [curr_state.player2_pawns["M1"][0], "M"],
            "agent2_medium2": [curr_state.player2_pawns["M2"][0], "M"],
            "agent2_small1": [curr_state.player2_pawns["S1"][0], "S"],
            "agent2_small2": [curr_state.player2_pawns["S2"][0], "S"]
        }
        location = action_to_direction[action[1]]
        for _, value in pawn_list.items():
            if np.array_equal(value[0], location):
                if size_cmp(value[1], action[0][0]) >= 0:
                    # print("ILLEGAL placement of pawn")
                    return False

        # finding current location
        curr_location = find_curr_location(curr_state, action[0], curr_state.turn)

        # check that the pawn is not under another pawn - relevant only to small and medium
        if action[0][0] != "B" and not np.array_equal(curr_location, not_on_board):
            for key, value in pawn_list.items():
                if np.array_equal(value[0], curr_location):
                    if size_cmp(value[1], action[0][0]) > 0:
                        # print("ILLEGAL pawn selection")
                        return False
        return True
    # Counting call count
    if not get_neighbors_super.call_count:
        get_neighbors_super.call_count = 0
    get_neighbors_super.call_count += 1
    
    # Checking possible neighbors according to the current turn (neglecting smaller ones on earlier turns)
    if get_neighbors_super.call_count <= 3:
        pawns = ["B1", "B2", "M1"]
    elif get_neighbors_super.call_count <= 4:
        pawns = ["B1", "B2", "M1", "M2"]
    elif get_neighbors_super.call_count <= 6:
        pawns = ["B1", "B2", "M1", "M2", "S1"]
    else:
        pawns = ["B1", "B2", "M1", "M2", "S1", "S2"]
    # pawns = ["B1", "B2", "M1", "M2", "S1", "S2"]
        
    # TODO remember to check if dry_action returned None
    neighbor_list = []
    # all the pawns we can select
    # locations (it's just 0 to 8  so I will use a simple loop)
    for i in range(9):
        for pawn in pawns:
            next_state = gge.State()
            next_state.insert_copy(state)
            # tmp_neighbor = self.dry_step((pawn, i), state)
            action = (pawn, i)
            if not is_legal_step(action, state):
                continue
            if state.turn == 0:
                next_state.player1_pawns[action[0]] = (action_to_direction[action[1]], action[0][0])
            else:
                next_state.player2_pawns[action[0]] = (action_to_direction[action[1]], action[0][0])

            next_state.turn = (next_state.turn + 1) % 2
            neighbor_list.append((action, next_state))

    return neighbor_list

get_neighbors_super.call_count = 0

# @functools.cache
def get_positional_value(pawn_tuple):
    # row 0, 1, 2, col 0, 1, 2, diag 0, 1
    trios = [[], [], [], [], [], [], [], []]
    for pawn in pawn_tuple:
        if np.array_equal(pawn[0], not_on_board):
            continue
        (x, y) = pawn[0]
        value = size_to_value[pawn[1]].value
        trios[x].append(value)
        trios[y + 3].append(value)
        if x == y:
            trios[6].append(value)
        if x == (2 - y):
            trios[7].append(value)

    return sum([sum(trio) * (len(trio) ** 3) for trio in trios])

@functools.cache
def get_positional_value_2(pawn_tuple):
    # row 0, 1, 2, col 0, 1, 2, diag 0, 1
    trios = [[], [], [], [], [], [], [], []]
    for pawn in pawn_tuple:
        if np.array_equal(pawn[0], not_on_board):
            continue
        (x, y) = pawn[0]
        value = size_to_value[pawn[1]].value
        trios[x].append(value)
        trios[y + 3].append(value)
        if x == y:
            trios[6].append(value)
        if x == (2 - y):
            trios[7].append(value)

    return sum([sum(trio) * (len(trio) ** 3) for trio in trios])


def smart_heuristic(state: gge.State, agent_id):
    # # Parameters and consts
    # # power_superparam = 1  # By increasing this we're making sure the agent favors combos (eg. 2 in a row)
    # # eaten_weight_superparam = 1  # controls the weighting of eating vs getting combos
    heuristic_value = 0
    position_value = 0
    eaten_value = 0

    pawns = [[value for key, value in state.player1_pawns.items() if not is_hidden(state, 0, key)],
             [value for key, value in state.player2_pawns.items()
              if not is_hidden(state, 1, key)]
             ]
    position_value = get_positional_value(
        pawns[agent_id]) - get_positional_value(pawns[1 - agent_id])

    # Sum values of enemy eaten (hidden) goblins
    player_pawns = [state.player1_pawns, state.player2_pawns]
    for key, value in player_pawns[1-agent_id].items():
        if is_hidden(state, 1-agent_id, key):
            eaten_value += size_to_value[value[1]].value

    heuristic_value = position_value + 2 * eaten_value
    return heuristic_value


def smart_heuristic_2(state: gge.State, agent_id):
    # # Parameters and consts
    # # power_superparam = 1  # By increasing this we're making sure the agent favors combos (eg. 2 in a row)
    # # eaten_weight_superparam = 1  # controls the weighting of eating vs getting combos
    heuristic_value = 0
    position_value = 0
    eaten_value = 0

    pawns = [tuple((tuple(value[0]), value[1]) for key, value in state.player1_pawns.items() if not is_hidden(state, 0, key)),
             tuple((tuple(value[0]), value[1]) for key, value in state.player2_pawns.items()
              if not is_hidden(state, 1, key))
             ]
    position_value = get_positional_value_2(
        pawns[agent_id]) - get_positional_value_2(pawns[1 - agent_id])

    # Sum values of enemy eaten (hidden) goblins
    player_pawns = [state.player1_pawns, state.player2_pawns]
    for key, value in player_pawns[1-agent_id].items():
        if is_hidden(state, 1-agent_id, key):
            eaten_value += size_to_value[value[1]].value

    heuristic_value = position_value + 2 * eaten_value
    return heuristic_value


super_agent_1 = SuperAgentFactory(smart_heuristic)
super_agent_2 = SuperAgentFactory(smart_heuristic_2)

# super_agent_2.super_agent(initial_state, 0, 3)
# env = gge.GridWorldEnv('human')
# env.reset()
# initial_state = env.get_state()
# super_agent_1.super_agent(initial_state, 0, 3)

# print(get_positional_value_2.cache_info())

# game(alpha_beta, super_agent_instance.super_agent, 3)
print(game(super_agent_1.super_agent, super_agent_2.super_agent, 3))
# print(game(super_agent_2.super_agent, super_agent_1.super_agent, 20))

print(get_positional_value_2.cache_info())
