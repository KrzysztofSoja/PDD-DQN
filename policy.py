import numpy as np


class AbstractPolicy:

    def __call__(self, q_values: np.ndarray) -> int:
        """ Return number of action. """
        raise NotImplementedError

    def update(self):
        pass


class Greedy(AbstractPolicy):

    def __call__(self, q_values: np.ndarray) -> int:
        return np.argmax(q_values)
