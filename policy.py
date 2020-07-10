import numpy as np
import random as rand


class AbstractPolicy:

    def __call__(self, q_values: np.ndarray) -> np.array:
        """ Return number of action. """
        raise NotImplementedError

    def update(self):
        pass

    def __str__(self):
        raise NotImplementedError


class Greedy(AbstractPolicy):

    def __call__(self, q_values: np.ndarray) -> np.array:
        return np.argmax(q_values)

    def __str__(self):
        return "\n" \
               "Greedy policy.\n\n"


class EpsGreedy(AbstractPolicy):

    def __init__(self, **kwargs):
        self.initial_eps = kwargs['initial_eps']
        self.eps = kwargs['initial_eps']
        self.eps_update = kwargs['eps_update']
        self.eps_min = kwargs['eps_min']

    def __call__(self, q_values: np.ndarray) -> np.array:
        if self.eps > rand.random():
            return np.random.randint(0, q_values.shape[0])
        else:
            return np.argmax(q_values)

    def __str__(self):
        return "\n" \
               "Eps-greedy policy. \n" \
               "Initial epsilon value: " + str(self.initial_eps) + "\n" + \
               "Epsilon update vaulue: " + str(self.eps_update) + "\n" + \
               "Minimal epsilon value: " + str(self.eps_min) + "\n\n"

    def update(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_update
        else:
            self.eps = self.eps_min
