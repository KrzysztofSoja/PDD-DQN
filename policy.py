import numpy as np


class AbstractPolicy:

    def __init__(self):
        pass

    def __call__(self, q_values: np.ndarray) -> int:
        """ Return number of action. """
        pass

    def update(self):
        pass
