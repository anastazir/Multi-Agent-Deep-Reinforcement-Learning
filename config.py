N_AGENTS          = 1
GRID_SIZE         = 5
CUSTOM_MAP        =["SFFFF",
                    "FFFFF",
                    "FFFFF",
                    "FFFFF",
                    "FFFFG"]
# GRID_SIZE = 10
# CUSTOM_MAP = ["SFFFFFFFFF",
#               "FFFFFFFFFF",
#               "FFFFFFFFFF",
#               "FFFFFFFFFF",
#               "FFFFFFFFFF",
#               "FFFFFFFFFF",
#               "FFFFFFFFFF",
#               "FFFFFFFFFF",
#               "FFFFFFFFFF",
#               "FFFFFFFFFG"]

POSSIBLE_ACTIONS  = ['U', 'D', 'L', 'R']
POSSIBLE_ACTIONS_NUM = [0, 1 ,2, 3]
STATE_SIZE        = GRID_SIZE*GRID_SIZE 

PLAYER_POS        = [(0,0)]
ENEMY_POS         = [(4,4)]
OBSTRUCTION_P0S   = [(1, 4), (2, 2), (4, 0)]

BATCH_SIZE        = 32
REPLAY_MEMORY_LEN = 2000
EPOCHS            = 1

ACTION_SIZE       = len(POSSIBLE_ACTIONS)

LEARNING_RATE     = 0.001 # 5e-05
GAMMA             = 0.95  # 0.6
DECAY_RATE        = 0.05
MAX_EPSILON       = 1.0
MIN_EPSILON       = 0.01

REPLAY_STEPS      = 6
TIME_STEPS        = 50
EPISODES          = 75

POSITIVE_REWARD   = 10
PROXIMITY_REWARD  = 0
NEGATIVE_REWARD   = -1
PENALTY_REWARD    = -2

HIDDEN_LAYER_01   = 256
HIDDEN_LAYER_02   = 128
