import unittest
import numpy as np

from policy import EpsGreedy


class EpsGreedyTest(unittest.TestCase):

    def test_exploitation_is_growing(self):
        bandit = np.arange(-10, 10, dtype=np.float)
        policy = EpsGreedy(start_eps=1.0, eps_min=0.0, eps_update=0.98)

        gains = []
        for _ in range(100):
            gain = np.average([bandit[policy(bandit)] for _ in range(1000)])
            gains.append(gain)
            policy.update()

        smaller_counter = 0
        greater_counter = 0
        for idx, _ in enumerate(gains[:-1]):
            if gains[idx] > gains[idx + 1]:
                smaller_counter += 1
            elif gains[idx] < gains[idx + 1]:
                greater_counter += 1

        self.assertGreater(greater_counter, smaller_counter)
