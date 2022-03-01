import time
import random
import numpy as np
from config                      import *
from collections                 import deque
from tensorflow.keras            import Model, Sequential
from tensorflow.keras.layers     import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models     import load_model

class Agent:
    MAX_EPSILON = MAX_EPSILON
    MIN_EPSILON = MIN_EPSILON
    decay_rate = DECAY_RATE
    def __init__(self, index, pos):

        # Initialize atributes
        self._state_size = GRID_SIZE*GRID_SIZE
        self._action_size = 5
        self._optimizer = Adam(learning_rate=LEARNING_RATE)
        self.index = index
        self.terminal = False
        self.replay_memory_len = REPLAY_MEMORY_LEN
        self.expirience_replay = deque(maxlen=self.replay_memory_len)
        self.x = pos[0]
        self.y = pos[1]
        self.batch_size = BATCH_SIZE
        self.n_agents = N_AGENTS
        # Initialize discount and exploration rate
        self.gamma = GAMMA
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
        model.add(Dense(ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        if self.terminal:
            self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, states, possibleActions):
        if self.terminal:
            return 0
        if np.random.rand() <= self.epsilon or self.batch_size > len(self.expirience_replay):
            # print("random action taken")
            return possibleActions[np.random.choice([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4], size=1, replace=False)[0]]
        states = np.array(states)
        states = states.ravel()
        q_values = self.q_network.predict(states)
        action = np.argmax(q_values[0])
        return possibleActions[action]

    def retrain(self):
        if len(self.expirience_replay) <= self.batch_size:
            return
        minibatch = random.sample(self.expirience_replay, self.batch_size)

        for state, action, reward, next_state, terminated in minibatch:

            target = self.q_network.predict(state)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)
        # print("len of memory" ,len(self.expirience_replay), " and epsilon is ", self.epsilon)

    def set_pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]
    
    def decay_epsilon(self, episode):
        # slowly decrease Epsilon based on our experience
        self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * \
        np.exp(-self.decay_rate*episode)

    def save_model(self):
        self.q_network.save(f"0{self.index}_{self.n_agents}.h5")

    def load_model(self):
        self.q_network = load_model(f"0{self.index}_{self.n_agents}.h5")

    def return_coordinates(self):
        return (self.x, self.y)