import random
import numpy as np
import tensorflow as tf
from config                      import *
from collections                 import deque
from tensorflow.keras            import Sequential
from tensorflow.keras.layers     import Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models     import load_model

class Agent:

    decay_rate = DECAY_RATE
    def __init__(self, index, pos, test = False, type = "sticky"):
        self.test = test
        self.state_size = GRID_SIZE*GRID_SIZE
        self.action_size = 5
        self._optimizer = RMSprop(learning_rate=LEARNING_RATE)
        self.index = index
        self.terminal = False
        self.replay_memory_len = REPLAY_MEMORY_LEN
        self.expirience_replay = deque(maxlen=self.replay_memory_len)
        self.x = pos[0]
        self.y = pos[1]
        self.batch_size = BATCH_SIZE
        self.n_agents = N_AGENTS
        self.gamma = GAMMA
        self.epsilon = MAX_EPSILON if not test else MIN_EPSILON
        self.type = type
        self.model = self._build_compile_model()

    def store(self, new_state, reward, done, state, action):
        if self.terminal:
            return
        self.expirience_replay.append((new_state, reward, done, state, action))

    def _build_compile_model(self):
        # model = Sequential()
        # model.add(Dense(32, input_shape=(GRID_SIZE, GRID_SIZE,1), activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(ACTION_SIZE, activation='linear'))
        # model.compile(loss='mse', optimizer=self._optimizer)
        # return model
        model=tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (2,2), input_shape=(GRID_SIZE*10, GRID_SIZE*10, 1), activation='relu'),
            # tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D(128, (2,2),  activation='relu'),
            # tf.keras.layers.MaxPooling2D(2,2),

            # tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
            # tf.keras.layers.MaxPooling2D(2,2),
            
            # tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
            # tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(32,activation='relu'),

            tf.keras.layers.Dense(len(POSSIBLE_ACTIONS),activation='softmax')
        ])
        model.compile(optimizer=self._optimizer,loss="mse",metrics=["mean_squared_error"])
        return model

    def act(self, state, possibleActions):
        if self.terminal:
            return 'S'
        if np.random.rand() < self.epsilon and not self.test:
            print("random action")
            return possibleActions[np.random.choice(POSSIBLE_ACTIONS_NUM, size=1, replace=False)[0]]
        return possibleActions[np.argmax(self.model.predict(state))]


    def retrain(self, episode):
        if len(self.expirience_replay) <= BATCH_SIZE:
            return
        minibatch=random.sample(self.expirience_replay, BATCH_SIZE)
        for new_state, reward, done, state, action in minibatch:
            new_state = new_state/255.
            state = state/255.
            # print(new_state.shape, reward, done, state.shape, action)
            target= reward
            if not done:
                target=reward + self.gamma* np.amax(self.model.predict(new_state)[0])
            target_f= self.model.predict(state)
            # print("targetf", target_f)
            # print("target", target)
            # print("action", action)
            target_f[0][action]= target
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