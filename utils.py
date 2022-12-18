import time

import Gobblet_Gobblers_Env as gge
import submission


def game(agent_1, agent_2, time_limit):
    s = gge.State()
    env = gge.GridWorldEnv('human')
    winner = None
    env.reset()
    env.render()
    start_time = 0
    end_time = 0
    steps_per_game = 0
    while winner is None:
        # state = env.get_state()
        # pprint(state.player1_pawns)
        # print(state.player2_pawns)
        # input()
        if env.s.turn == 0:
            print("player 0")
            start_time = time.time()
            chosen_step = agent_1(env.get_state(), 0, time_limit)
            end_time = time.time()
            if chosen_step is None:
                continue
            action = chosen_step[0], int(chosen_step[1])
            # if (end_time - start_time) > time_limit and (agent_1_str in ["minimax", "alpha_beta", "expectimax"]):
                # raise RuntimeError("Agent used too much time!")
            env.step(action)
            env.render()
            steps_per_game += 1
            print("time for step was", end_time - start_time)
        else:
            print("player 1")
            start_time = time.time()
            chosen_step = agent_2(env.get_state(), 1, time_limit)
            end_time = time.time()
            if chosen_step is None:
                continue
            action = chosen_step[0], int(chosen_step[1])
            # if (end_time - start_time) > time_limit and (agent_2_str in ["minimax", "alpha_beta", "expectimax"]):
                # raise RuntimeError("Agent used too much time!")
            env.step(action)
            env.render()
            steps_per_game += 1
            print("time for step was", end_time - start_time)

        s = env.get_state()
        winner = gge.is_final_state(env.s)
        if steps_per_game >= 300:
            winner = 0
    if winner == 0:
        print("tie")
    else:
        print("winner is:", winner)
    return winner
