import time
import random
import numpy as np
import math
from IPython.display import clear_output
from config import *
class Enviroment:

    def __init__(self, initial_states = [], enemy_states = [], type = "stick") -> None:
        self.possibleActions = POSSIBLE_ACTIONS
        self.initial_states = initial_states
        self.grid_size = GRID_SIZE
        self.m = self.grid_size
        self.stateSpacePlus = [i for i in range(self.grid_size*self.grid_size)]
        self.actionSpace = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1, 'S': 0}
        self.agents_state = initial_states
        self.enemy_states = enemy_states
        self.n_agents = N_AGENTS
        self.type = type

    def step(self, actions):
        new_states = []
        rewards = []
        terminal = []

        i = 0
        for action in actions:
            resultingState = self.agents_state[i] + self.actionSpace[action]

            if self.offGridMove(resultingState, self.agents_state[i]):
                new_states.append(self.agents_state[i])
                rewards.append(PENALTY_REWARD)
                if self.isTerminalState(self.agents_state[i]):
                    terminal.append(True)
                else:
                    terminal.append(False)
            else:
                new_states.append(resultingState)
                rewards.append(self.give_reward(resultingState))
                if self.isTerminalState(resultingState):
                    terminal.append(True)
                else:
                    terminal.append(False)
        self.agents_state = new_states
        return [new_states, rewards, terminal]         

    def render(self, clear = False):
        print('--------------------------------------------')

        for i in range(self.m):
            for j in range(self.m):
                if self.return_state(i, j) in self.enemy_states and\
                      self.return_state(i, j) in self.agents_state:
                    print("O", end='\t')
                elif self.return_state(i, j) in self.agents_state:
                    print('P', end='\t')
                elif self.return_state(i, j) in self.enemy_states:
                    print('X', end='\t')
                else:
                    print('-', end='\t')
            print('\n')

        print('--------------------------------------------')
        if clear:
            time.sleep(0.05)
            clear_output(wait = True)

    def actionSpaceSample(self):
        return [np.random.choice(self.possibleActions) for _ in range(self.n_agents)]

    def reset(self):
        self.agents_state = self.initial_states
        if self.type == "random":
            self.enemy_states = [random.randint(4, GRID_SIZE*GRID_SIZE - 1)  for _ in range(0, N_AGENTS)]
        return [self.initial_states, self.enemy_states]

    def isTerminalState(self, state):
        if state in self.enemy_states:
            return True
        else:
            return False

    def offGridMove(self, newState, oldState):
        if newState not in self.stateSpacePlus:
            return True
        elif oldState % self.m == 0 and newState  % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False

    def decode_state(self, state):
        return int(state/self.grid_size), state%self.grid_size

    def return_state(self, row, col):
        return row*self.m + col 

    def give_reward(self, state):
        # print("state is ", state)
        state_pos = self.decode_state(state)
        if state in self.enemy_states:
            return POSITIVE_REWARD
        # print("state_pos ,", state_pos)
        enemy_positions = [self.decode_state(state) for state in self.enemy_states]
        distances = []
        # print("enemy_positions, ", enemy_positions)
        for enemy_pos in enemy_positions:
            distances.append(math.sqrt((enemy_pos[0] - state_pos[0])**2 + (enemy_pos[1] - state_pos[1])**2))
        # print("distances", distances)
        if min(distances) <2:
            return PROXIMITY_REWARD
        else:
            return NEGATIVE_REWARD

    def status(self):
        return [self.agents_state, self.enemy_states]