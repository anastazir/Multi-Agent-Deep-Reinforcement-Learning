import numpy as np
from agent import Agent
from config import *

possibleActions = ['U', 'D', 'L', 'R' , 'S']

action_space_dict = {
    "U" : 1,
    "D" : 2,
    "L" : 3,
    "R" : 4,
    "S" : 0
}

class SuperAgent:
    def __init__(self):
        self.n_agents = N_AGENTS
        self.batch_size = BATCH_SIZE
        self.num_col= GRID_SIZE
        self.agent_positions = PLAYER_POS[: self.n_agents]
        self.enemy_positions = ENEMY_POS[: self.n_agents]
        self.all_agents = [Agent(index, self.agent_positions[index],\
                           self.batch_size) for index in range(N_AGENTS)]
        self.actions = []
        self.agents_state = []
        self.enemy_states = [self.state_encode(enemy_pos[0], enemy_pos[1])\
                            for enemy_pos in self.enemy_positions]

    def take_actions(self, states):
        self.actions = []
        for agent in self.all_agents:
            self.actions.append(agent.act(states, possibleActions))
        return self.actions

    def store_memory(self, states, next_states, rewards, done):
        states = np.concatenate((states, self.enemy_states)).ravel()
        next_states = np.concatenate((next_states,self.enemy_states)).ravel()
        for agent in self.all_agents:
            agent.set_pos(self.decode_state(next_states[agent.index]))
            agent.store(states, action_space_dict[self.actions[agent.index]], rewards[agent.index],\
            next_states, done[agent.index])

            if done[agent.index] == True:
                agent.terminal = True

    def retrain_agents(self):
        for agent in self.all_agents:
            if agent.terminal:
                agent.alighn_target_model()
            agent.retrain()

    def reset_agents(self):
        for agent in self.all_agents:
            agent.terminal = False
            agent.set_pos(self.agent_positions[agent.index])

    def reduce_exploration(self, episode):
        for agent in self.all_agents:
            agent.decay_epsilon(episode)

    def return_agents_state(self):
        self.agents_state = []
        for agent in self.all_agents:
            self.agents_state.append(self.state_encode(agent.x, agent.y))

    def return_enemies_state(self):
        self.enemy_states = []
        for enemy_pos in self.enemy_positions:
            self.enemy_states.append(self.state_encode(enemy_pos[0], enemy_pos[1]))

    def decode_state(self, state_num):
        return int(state_num/self.num_col), state_num%self.num_col

    def state_encode(self, row,col):
        return row*self.num_col + col

    def save_agents(self):
        for agent in self.all_agents:
            agent.save_model()

    def load_model(self):
        for agent in self.all_agents:
            agent.load_model()

    def show_info(self):
        avg_epsilon = sum([agent.epsilon for agent in self.all_agents])/self.n_agents
        avg_memory = sum([len(agent.expirience_replay) for agent in self.all_agents])/self.n_agents
        print("avg_epsilon is ", avg_epsilon, "avg_memory is ", avg_memory)