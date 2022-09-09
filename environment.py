import time
import random
import numpy as np
import math
from IPython.display import clear_output
from config import *
from landmark import Landmark
import numpy as np
import matplotlib.pyplot as plt
import random

class Environment:

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
        self.obstruction_states = [self.return_state(i, j) for i, j in OBSTRUCTION_P0S]
        self.reset_map = self.create_map()
        self.map_image = self.create_map()

    def create_map(self):
        black_env = np.zeros((GRID_SIZE*10,GRID_SIZE*10), np.uint8)
        for state in self.enemy_states:
            y, x = self.decode_state(state)
            x = x*10
            y = y*10
            black_env[y:y+10, x:x+10] = 10
        for state in self.agents_state:
            y, x = self.decode_state(state)
            x = x*10
            y = y*10
            black_env[y:y+10, x:x+10] = 100
        if self.type == "obstruction":
            for state in self.obstruction_states:
                y, x = self.decode_state(state)
                x = x*10
                y = y*10
                black_env[y:y+10, x:x+10] = 200

        black_env = np.expand_dims(black_env, axis=2)
        black_env = black_env.reshape(-1, GRID_SIZE*10, GRID_SIZE*10, 1)
        return black_env

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
        self.map_image = self.create_map()
        return [self.map_image, new_states, rewards, terminal]

    def show_image(self):
        plt.imshow(self.map_image[0])
        plt.show()

    def render(self):
        print('--------------------------------------------')

        for i in range(self.m):
            for j in range(self.m):
                if self.return_state(i, j) in self.enemy_states and \
                      self.return_state(i, j) in self.agents_state:
                    print("O", end='\t')
                elif self.return_state(i, j) in self.agents_state:
                    print('P', end='\t')
                elif self.return_state(i, j) in self.enemy_states:
                    print('X', end='\t')
                elif self.return_state(i, j) in self.obstruction_states and self.type == "obstruction":
                    print("■", end='\t')
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
            self.agents_state = [random.randint(0, 10)  for _ in range(0, N_AGENTS)]
            self.initial_states = self.agents_state
        elif self.type == "random_enemy":
            self.enemy_states = [random.randint(11, 24)  for _ in range(0, N_AGENTS)]
            for landmark in self.all_landmarks:
                landmark.state = self.enemy_states[landmark.index]
                landmark.x, landmark.y = self.decode_state(self.enemy_states[landmark.index])
            self.agents_state = self.initial_states
        else: 
            self.agents_state = self.initial_states
        self.map_image = self.create_map()
        return [self.map_image, self.agents_state, self.enemy_states]

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
        elif newState in self.obstruction_states and self.type == "obstruction":
            return True
        else:
            return False

    def decode_state(self, state):
        return int(state/self.grid_size), int(state%self.grid_size)

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