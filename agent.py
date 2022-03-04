import random
import numpy as np
from config                      import *
from collections                 import deque
from tensorflow.keras            import Model, Sequential
from tensorflow.keras.layers     import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models     import load_model

class Agent:

    decay_rate = DECAY_RATE
    def __init__(self, index, pos, test = False):
        self.test = test
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
        self.epsilon = MAX_EPSILON if not test else MIN_EPSILON

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
        model.add(Dense(HIDDEN_LAYER_01, activation='relu'))
        model.add(Dense(HIDDEN_LAYER_02, activation='relu'))
        model.add(Dense(ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.save_model()
        if self.terminal:
            self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, states, possibleActions):
        if self.terminal:
            return 0
        if np.random.rand() <= self.epsilon or self.batch_size > len(self.expirience_replay):
            return possibleActions[np.random.choice(POSSIBLE_ACTIONS_NUM, size=1, replace=False)[0]]
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
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
        np.exp(-self.decay_rate*episode)

    def save_model(self):
        self.q_network.save(f"./saved_models/0{self.index}_{N_AGENTS}_{GRID_SIZE}_{REPLAY_MEMORY_LEN}.h5")

    def load_model(self):
        self.q_network = load_model(f"./saved_models/0{self.index}_{N_AGENTS}_{GRID_SIZE}_{REPLAY_MEMORY_LEN}.h5")

    def return_coordinates(self):
        return (self.x, self.y)

    def print_summary(self):
        print("q_network summary")
        self.q_network.summary()
        print("--------------")
        print("target_network")
        self.target_network.summary()