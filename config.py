N_AGENTS          = 1
GRID_SIZE         = 5

PLAYER_POS        = [(0,2),(1,5),(0,6),(0,3)]
ENEMY_POS         = [(4,4),(3,2),(6,6),(6,1)]

BATCH_SIZE        = 256
REPLAY_MEMORY_LEN = 10000

ACTION_SIZE       = 5

LEARNING_RATE     = 0.8   # 5e-05
GAMMA             = 0.95  # 0.6
DECAY_RATE        = 0.05
MAX_EPSILON       = 1.0
MIN_EPSILON       = 0.01

EPOCHS            = 1
REPLAY_STEPS      = 8
TIME_STEPS        = 99
EPISODES          = 200
NUM_NODES         = 256

POSITIVE_REWARD   = 10
PROXIMITY_REWARD  = 1
NEGATIVE_REWARD   = -1
NO_REWARD         = 0