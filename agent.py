import time
import random
import numpy as np
from collections                 import deque
from tensorflow.keras            import Model, Sequential
from tensorflow.keras.layers     import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models     import load_model

class Agent:
    MAX_EPSILON = 1.0
    MIN_EPSILON = 0.01
    decay_rate = 1e-0
    def __init__(self, index, pos, batch_size = 32, replay_memory_len = 2000, n_agents = 4):

        # Initialize atributes
        self._state_size = 100
        self._action_size = 5
        self._optimizer = Adam(learning_rate=0.01)
        self.index = index
        self.terminal = False
        self.replay_memory_len = replay_memory_len
        self.expirience_replay = deque(maxlen=self.replay_memory_len)
        self.x = pos[0]
        self.y = pos[1]
        self.batch_size = batch_size
        self.n_agents = n_agents
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
        if self.terminal:
            self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, states, possibleActions):
        if self.terminal:
            return 0
        if np.random.rand() <= self.epsilon or self.batch_size > len(self.expirience_replay):
            return possibleActions[np.random.choice([0,1,2,3,4], size=1, replace=False)[0]]
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