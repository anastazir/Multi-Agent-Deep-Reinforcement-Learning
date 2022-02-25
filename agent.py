import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import time
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

class Agent:
    MAX_EPSILON = 1.0
    MIN_EPSILON = 0.01
    decay_rate = 1e-0
    def __init__(self, index, pos):

        # Initialize atributes
        self._state_size = 100
        self._action_size = 5
        self._optimizer = Adam(learning_rate=0.01)
        self.index = index
        self.terminal = False
        self.expirience_replay = deque(maxlen=2000)
        self.x = pos[0]
        self.y = pos[1]

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = self.MAX_EPSILON

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))

    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, states, possibleActions):
        if self.terminal:
            return 0

        if np.random.rand() <= self.epsilon:
            return possibleActions[np.random.choice([0,1,2,3,4], size=1, replace=False)]

        q_values = self.q_network.predict(states[self.index])
        action = np.argmax(q_values[0])
        return possibleActions[action]

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:

            target = self.q_network.predict(state)

            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)

    def set_pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]
    
    def decay_epsilon(self, episode):
        # slowly decrease Epsilon based on our experience
        self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * \
        np.exp(-self.decay_rate*episode) 