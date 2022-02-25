from itertools import permutations
import numpy as np

class Enviroment:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    A = [UP, DOWN, LEFT, RIGHT, STAY]
    A_DIFF = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    num_col = 10
    action_list=list(permutations(range(0,5)))
    current_state = []

    def __init__(self, m = 10, n = 10, initial_states = [], enemy_states = []) -> None:
        self.possibleActions = ['U', 'D', 'L', 'R', 'S']
        self.initial_states = initial_states
        self.m = m
        self.n = n
        self.stateSpacePlus = [i for i in range(m*n)]
        self.actionSpace = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1, 'S': 0}
        self.possibleActions = ['U', 'D', 'L', 'R']
        self.agents_state = initial_states
        self.enemy_states = enemy_states

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

    def render(self):
        pass


    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)


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