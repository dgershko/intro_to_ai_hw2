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

class size_to_value(enum.Enum):
    B = 3
    M = 2
    S = 1

@functools.cache
def trio_value(a = 0, b = 0, c = 0):
    trio = [a,b,c]
    return len(trio) * sum(trio)

arr = np.array([1,2,0])


# @functools.cache
def get_positional_value(pawn_tuple):
    trios = [[], [], [], [], [], [], [], []] # row 0, 1, 2, col 0, 1, 2, diag 0, 1
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

def get_positional_value_2(pawn_tuple):
    trios = [[], [], [], [], [], [], [], []] # row 0, 1, 2, col 0, 1, 2, diag 0, 1
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
             [value for key, value in state.player2_pawns.items() if not is_hidden(state, 1, key)]
    ]
    position_value = get_positional_value(pawns[agent_id]) - get_positional_value(pawns[1 - agent_id])
    
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

    pawns = [[value for key, value in state.player1_pawns.items() if not is_hidden(state, 0, key)],
             [value for key, value in state.player2_pawns.items() if not is_hidden(state, 1, key)]
    ]
    position_value = get_positional_value_2(pawns[agent_id]) - 5 * get_positional_value_2(pawns[1 - agent_id])

    # Sum values of enemy eaten (hidden) goblins
    player_pawns = [state.player1_pawns, state.player2_pawns]
    for key, value in player_pawns[1-agent_id].items():
        if is_hidden(state, 1-agent_id, key):
            eaten_value += size_to_value[value[1]].value
    
    heuristic_value = position_value + 2 * eaten_value
    return heuristic_value

super_agent_1 = SuperAgentFactory(smart_heuristic)
super_agent_2 = SuperAgentFactory(smart_heuristic_2)

# game(alpha_beta, super_agent_instance.super_agent, 3)
print(game(super_agent_1.super_agent, super_agent_2.super_agent, 80))
# print(game(super_agent_2.super_agent, super_agent_1.super_agent, 80))

