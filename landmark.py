from config import *

class Landmark:
    def __init__(self, index, state, is_captured=False):
        self.index = index
        self.state = state
        self.is_captured = is_captured
        self.x, self.y = self.decode_state()

    def decode_state(self):
        return int(self.state / GRID_SIZE), self.state % GRID_SIZE

    def captured(self):
        print("landmark captured", self.index)
        self.is_captured = True

    def reset(self):
        self.is_captured = False