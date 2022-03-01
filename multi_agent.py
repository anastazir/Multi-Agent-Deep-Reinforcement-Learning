import  time
import  numpy as np
from    config          import *
from    agent           import Agent
from    enviroment      import Enviroment
from    IPython.display import clear_output
from    matplotlib      import pyplot as plt

grid_size = GRID_SIZE
num_col = grid_size

possibleActions = ['U', 'D', 'L', 'R', 'S']

action_space_dict = {
    "U" : 1,
    "D" : 2,
    "L" : 3,
    "R" : 4,
    "S" : 0
}

n_agents          = N_AGENTS
allplayerpos      = PLAYER_POS[: n_agents]
enemy_list_pos    = ENEMY_POS[: n_agents]
batch_size        = BATCH_SIZE
replay_memory_len = REPLAY_MEMORY_LEN

def decode_state(state_num):
    return int(state_num/num_col), state_num%num_col

def state_encode(row,col):
    return row*num_col + col 

def run():
    total_step = 0
    rewards_list = []
    timesteps_list = []

    for episode in range(EPISODES):
        print("Episode number: " + episode)

        reward_all = 0
        time_step = 0
        i = 0

        for agent in all_agents:
            agent.terminal = False
        
        states = env.reset()
        states = np.concatenate((states,enemy_states)).ravel()

        for agent in all_agents:
            agent.set_pos(allplayerpos[i])
            i = i + 1

        done = [False for _ in range(n_agents)]

        while not any(done):

            env.render(clear=True)
            actions = []

            for agent in all_agents:
                actions.append(agent.act(states, possibleActions))

            next_states, rewards, done = env.step(actions)
            for agent in all_agents:

                next_states = np.array(next_states)
                next_states = next_states.ravel()

                agent.set_pos(decode_state(next_states[agent.index]))
                agent.store(np.concatenate((states,enemy_states)).ravel(), action_space_dict[actions[agent.index]], \
                rewards[agent.index], np.concatenate((next_states,enemy_states)).ravel(), done[agent.index])

                if done[agent.index] == True:
                    agent.terminal = True
                    print("agent reached landmark")

            total_step += 1
            time_step += 1
            states = next_states
            reward_all += sum(rewards)

            for agent in all_agents:
                if agent.terminal:
                    agent.alighn_target_model()
                if time_step % REPLAY_STEPS == 0:
                    agent.retrain()

        for agent in all_agents:
            agent.decay_epsilon(episode)

        rewards_list.append(reward_all)
        timesteps_list.append(time_step)

all_agents= []

for index in range(n_agents):

    all_agents.append(Agent(index, allplayerpos[index], batch_size, replay_memory_len))


initial_states = []
for agent in all_agents:
    initial_states.append(state_encode(agent.x, agent.y))

enemy_states = []
for enemy_pos in enemy_list_pos:
    enemy_states.append(state_encode(enemy_pos[0], enemy_pos[1]))

env = Enviroment(grid_size = grid_size, initial_states = initial_states, enemy_states = enemy_states, n_agents = n_agents)