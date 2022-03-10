import time
import random
import numpy as np
import math
from config import *
from landmark import Landmark
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
        self.all_landmarks = [Landmark(index, enemy_states[index]) for index in range(len(enemy_states))]

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
                terminal.append(False)
            else:
                new_states.append(resultingState)
                rewards.append(self.give_reward(resultingState))
                if rewards[-1] == POSITIVE_REWARD:
                    terminal.append(True)
                else:
                    terminal.append(False)
            i+=1
        self.agents_state = new_states
        return [new_states, rewards, terminal]

    def render(self):
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

    def actionSpaceSample(self):
        return [np.random.choice(self.possibleActions) for _ in range(self.n_agents)]

    def reset(self):
        for landmark in self.all_landmarks:
            landmark.reset()

        if self.type == "random":
            self.agents_state = [random.randint(0, 18)  for _ in range(0, N_AGENTS)]
            self.initial_states = self.agents_state
        else:
            self.agents_state = self.initial_states
        return [self.agents_state, self.enemy_states]

    def isTerminalState(self, state):
        if state in self.enemy_states:
            for landmark in self.all_landmarks:
                if landmark.state == state and not landmark.is_captured:
                    landmark.captured()
                    return True
            return False
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
        state_pos = self.decode_state(state)
        if self.isTerminalState(state):
            return POSITIVE_REWARD
        distances = []
        for landmark in self.all_landmarks:
            if not landmark.is_captured:
                distances.append(math.sqrt((landmark.x - state_pos[0])**2 + (landmark.y - state_pos[1])**2))
        if not distances:
            print("No distances were found")
            return NEGATIVE_REWARD
        if min(distances) <2:
            return PROXIMITY_REWARD
        else:
            return NEGATIVE_REWARD

    def status(self):
        return [self.agents_state, self.enemy_states]