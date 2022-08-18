import random
import numpy as np
import tensorflow as tf
from config                      import *
from collections                 import deque
from tensorflow.keras            import Sequential
from tensorflow.keras.layers     import Dense
from tensorflow.keras.layers     import Dense, Activation, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models     import load_model

class Agent:

    def __init__(self, index, pos, test = False, type = "sticky"):
        self.test = test
        self.state_size = GRID_SIZE*GRID_SIZE
        self.action_size = 4
        self._optimizer = RMSprop(learning_rate=LEARNING_RATE)
        self.index = index
        self.terminal = False
        self.replay_memory_len = REPLAY_MEMORY_LEN
        self.expirience_replay = deque(maxlen=self.replay_memory_len)
        self.x = pos[0]
        self.y = pos[1]
        self.batch_size = BATCH_SIZE
        self.n_agents = N_AGENTS
        # self.gamma = GAMMA
        # self.epsilon = MAX_EPSILON if not test else MIN_EPSILON
        self.type = type
        self.model = self._build_compile_model()
        self.target_model = self._build_compile_model()
        self.target_model.set_weights(self.model.get_weights())
        # Hyperparameters
        self.gamma = 0.99           # Discount rate
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.1      # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.update_rate = 20
        

    def store(self, new_state, reward, done, state, action):
        if self.terminal:
            return
        self.expirience_replay.append((new_state, reward, done, state, action))

    def _build_compile_model(self):
        model = Sequential()
        
        # Conv Layers
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=(GRID_SIZE*10, GRID_SIZE*10, 1)))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=None, decay=0.0))
        return model

    def act(self, state, possibleActions):
        if self.terminal:
            return 'S'
        if np.random.rand() < self.epsilon and not self.test:
            print("random action")
            return possibleActions[np.random.choice(POSSIBLE_ACTIONS_NUM, size=1, replace=False)[0]]

        act_values = self.model.predict(state)
        
        return possibleActions[np.argmax(act_values[0])]  # Returns action using policy


    def retrain(self, episode):
        if len(self.expirience_replay) <= BATCH_SIZE:
            return
        minibatch=random.sample(self.expirience_replay, BATCH_SIZE)
        for next_state, reward, done, state, action in minibatch:

            if not done:
                max_action = np.argmax(self.model.predict(next_state)[0])
                target = (reward + self.gamma * self.target_model.predict(next_state)[0][max_action])
            else:
                target = reward

            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.model.predict(state)

            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target

            # 3. Use vectors in the objective computation
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > MIN_EPSILON:
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
            np.exp(-DECAY_RATE*episode)
        print("epsilon ", self.epsilon)

    def set_pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]

    def save_model(self):
        self.model.save(f"./saved_models/0{self.index}_{N_AGENTS}_{GRID_SIZE}_{REPLAY_MEMORY_LEN}_{TIME_STEPS}_{self.type}_image.h5")

    def load_model(self):
        self.model = load_model(f"./saved_models/0{self.index}_{N_AGENTS}_{GRID_SIZE}_{REPLAY_MEMORY_LEN}_{TIME_STEPS}_{self.type}_image.h5")

    def return_coordinates(self):
        return (self.x, self.y)

    def print_summary(self):
        print("model summary")
        self.model.summary()
        print("--------------")

    def update_target_model(self):
        print("update_target_model")
        self.target_model.set_weights(self.model.get_weights())