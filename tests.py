import numpy as np
from pprint import pprint
import enum
import Gobblet_Gobblers_Env as gge
import time
import math
import signal
import multiprocessing


from submission import alpha_beta, SuperAgentFactory
from utils import game

def heuristic(state: gge.State, agent_id):
    return 0

super_agent_instance = SuperAgentFactory(heuristic)

game(alpha_beta, super_agent_instance.super_agent, 3)


