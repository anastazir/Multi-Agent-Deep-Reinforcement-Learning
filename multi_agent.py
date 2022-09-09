import  time
import  numpy as np
from    config          import *
from    agent           import Agent
from    environment      import Environment
from    IPython.display import clear_output
from    matplotlib      import pyplot as plt
import  random

grid_size = GRID_SIZE
num_col = grid_size

possibleActions = POSSIBLE_ACTIONS

action_space_dict = {
    "U" : 0,
    "D" : 1,
    "L" : 2,
    "R" : 3,
    "S" : 4
}

n_agents          = N_AGENTS
allplayerpos      = PLAYER_POS[: n_agents]
enemy_list_pos    = ENEMY_POS[: n_agents]
batch_size        = BATCH_SIZE
replay_memory_len = REPLAY_MEMORY_LEN
type = "sticky"

def decode_state(state_num):
    return int(state_num/num_col), state_num%num_col

def state_encode(row,col):
    return row*num_col + col

all_agents = []
for i in range(0, N_AGENTS):
    all_agents.append(Agent(i, allplayerpos[i], type = type))


initial_states = []
for agent in all_agents:
    initial_states.append(state_encode(agent.x, agent.y))

enemy_states = []
for enemy_pos in enemy_list_pos:
    enemy_states.append(state_encode(enemy_pos[0], enemy_pos[1]))

env = Environment(initial_states = initial_states, enemy_states = enemy_states, type = type)

def run():
    total_step = 0
    rewards_list = []
    timesteps_list = []
    total_steps = 1
    for episode in range(1, EPISODES):
        print("Episode number: ", episode)

        reward_all = 0
        time_step = 1
        for agent in all_agents:
            agent.terminal = False
        
        [states, enemy_states] = env.reset()
        print("player states: ", states)
        for agent in all_agents:
            agent.set_pos(allplayerpos[agent.index])

        done = [False for _ in range(n_agents)]

        state_arr = np.zeros(STATE_SIZE)
        state_arr[states] = 1
        state_arr[enemy_states] = 2
        # state_arr[np.concatenate((states,enemy_states))] = 1
        old_states = np.reshape(state_arr, [1, STATE_SIZE])
        # print("old states are", old_states)

        while not all(done):

            # env.render(clear=True)
            actions = []
            for agent in all_agents:

                actions.append(agent.act(old_states, possibleActions))

            next_states, rewards, done = env.step(actions)

            for agent in all_agents:
                agent.set_pos(decode_state(next_states[agent.index]))

            nstate_arr = np.zeros(STATE_SIZE)
            nstate_arr[next_states] = 1
            nstate_arr[enemy_states] = 2
            # nstate_arr[np.concatenate((next_states,enemy_states))] = 1
            new_states = np.reshape(nstate_arr, [1, STATE_SIZE])
            # print("new states are", new_states)
            for agent in all_agents:
                agent.store(new_states, rewards[agent.index], \
                done[agent.index], old_states, action_space_dict[actions[agent.index]])

                if done[agent.index] == True:
                    agent.terminal = True
                    print("agent reached landmark--------------------------------")

            print("actions", actions)

            for agent in all_agents:
                print("agent epsilon ", agent.epsilon, "agent memory len",\
                    len(agent.expirience_replay), "steps ", time_step,\
                    "reward", rewards[0], "next state ", next_states[0], "agent position ", agent.return_coordinates())

            if time_step >= TIME_STEPS:
                print("max steps reached")
                break

            old_states = new_states

            total_step += 1
            time_step += 1
            total_steps+1
            reward_all += sum(rewards)

            if all(done):
                print("-----------------------------------all agents reached landmark--------------------------------")
                for agent in all_agents:
                    agent.save_model()
        print("total rewards", reward_all)
        for agent in all_agents:
            agent.retrain(episode)

        rewards_list.append(reward_all)
        timesteps_list.append(time_step)

run()
