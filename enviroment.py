import time
import numpy as np
from IPython.display import clear_output
class Enviroment:

    def __init__(self, m = 10, n = 10, initial_states = [], enemy_states = [], n_agents = 4) -> None:
        self.possibleActions = ['U', 'D', 'L', 'R', 'S']
        self.initial_states = initial_states
        self.m = m
        self.n = n
        self.stateSpacePlus = [i for i in range(m*n)]
        self.actionSpace = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1, 'S': 0}
        self.agents_state = initial_states
        self.enemy_states = enemy_states
        self.grid = np.zeros((m,n))
        self.n_agents = n_agents

    def step(self, actions):
        new_states = []
        rewards = []
        terminal = []

        i = 0
        for action in actions:
            resultingState = self.agents_state[i] + self.actionSpace[action]
            if  not self.isTerminalState(resultingState):
                rewards.append(-1)
                terminal.append(False)
            else:
                rewards.append(0)
                terminal.append(True)

            if not self.offGridMove(resultingState, self.agents_state[i]):
                new_states.append(resultingState)
            else:
                new_states.append(self.agents_state[i])
            i = i + 1
        self.agents_state = new_states
        return [new_states, rewards, terminal]

    def render(self, clear = False):
        print('--------------------------------------------')

        for i in range(self.m):
            for j in range(self.n):
                if self.return_state(i, j) in self.agents_state:
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
        return self.initial_states

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

    def return_state(self, row, col):
        return row*self.m + col 