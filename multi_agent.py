from agent import Agent
from Gridworld import Gridworld
from collections import deque
import numpy as np
import random
from enviroment import Enviroment
from IPython.display import clear_output
import pickle 
from matplotlib import pyplot as plt  
import time

grid_size = 10
num_col = grid_size
game_mode='random'

possibleActions = ['U', 'D', 'L', 'R']

action_space_dict = {
    "U" : 1,
    "D" : 2,
    "L" : 3,
    "R" : 4,
    "S" : 0
}

allplayerpos=[(0,2),(1,5),(0,6),(0,3)]

enemy_list_pos=[(7,5),(7,3),(6,6),(6,1)]
batch_size = 32
n_agents = 4
replay_memory_len = 2000

def decode_state(state_num):
    return int(state_num/num_col), state_num%num_col

def state_encode(row,col):
    return row*num_col + col 


def run():
    total_step = 0
    rewards_list = []
    timesteps_list = []
    max_score = -10000


    for episode in range(300):
        actions = []
        done = False
        reward_all = 0
        time_step = 0
        i = 0

        for agent in all_agents:
            agent.terminal = False
        
        states = env.reset()

        for agent in all_agents:
            agent.set_pos(allplayerpos[i])
            i = i + 1


        while not done and time_step < 200:

            for agent in all_agents:
                actions.append(agent.act(states, possibleActions))

            next_states, rewards, done = env.step(actions)
            for agent in all_agents:
                state = np.reshape(states[agent.index], [1, 1])
                new_state = np.reshape(next_states[agent.index], [1, 1])
                agent.set_pos(decode_state(next_states[agent.index]))
                agent.store(state, action_space_dict[actions[agent.index]], rewards[agent.index],\
                new_state, done[agent.index])

                if done[agent.index] == True:
                    agent.terminal = True

            total_step += 1
            time_step += 1
            states = next_states
            reward_all += sum(rewards)

        for agent in all_agents:
            agent.decay_epsilon(episode)

        rewards_list.append(reward_all)
        timesteps_list.append(time_step)

all_agents= []

for index in range(n_agents):

    all_agents.append(Agent(index, allplayerpos[index]))


initial_states = []
for agent in all_agents:
    initial_states.append(state_encode(agent.x, agent.y))

enemy_states = []
for enemy_pos in enemy_list_pos:
    enemy_states.append(state_encode(enemy_pos[0], enemy_pos[1]))

env = Enviroment(initial_states = initial_states, enemy_states = enemy_states, n_agents = n_agents)