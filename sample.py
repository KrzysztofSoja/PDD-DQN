import numpy as np


class Sample:

    def __init__(self, state: np.ndarray,
                 action: int,
                 next_state: np.ndarray,
                 reward: float,
                 next_sample: object = None):
        """ next_sample is reference to Sample class"""
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.next_sample = next_sample  # Uwaga na garbage collector!!!

    def __str__(self):
        return 's: ' + str(self.state) + ', a: ' + str(self.action) \
                + ', s\': ' + str(self.next_state) + ', r: ' + str(self.reward) \
                + ', done: ' + str(self.next_sample is None)

    def is_done(self):
        """ Is last state in game. """
        return self.next_sample is None
