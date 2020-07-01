import numpy as np

from agents.dqn import DQN
from sample import Sample
from typing import List


class DoubleDQN(DQN):

    def __init__(self, *args, **kwargs):
        super(DoubleDQN, self).__init__(*args, **kwargs)

    def _bellman_equation(self, batch: List[Sample]) -> np.ndarray:
        state = np.array([sample.state for sample in batch])
        q_values = self.current_model.predict(state)

        for idx in range(q_values.shape[0]):
            q_values[idx][batch[idx].action] = batch[idx].reward
            if not batch[idx].is_done():
                best_action_for_q_next = np.argmax(self.current_model.predict(batch[idx].next_state))
                q_next = self.model.predict(batch[idx].next_state)[0][best_action_for_q_next]
                q_values[idx][batch[idx].action] += self.gamma*q_next

        return q_values
