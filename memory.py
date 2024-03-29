import numpy as np
import random as rand

from sample import Sample
from typing import List
from collections import deque
from keras import Model
from data_structures.sumtree.sumtree import SumTree


class AbstractMemory:

    def __init__(self, maxlen: int):
        self.maxlen = maxlen

    def sample(self, batch_size: int) -> List[Sample]:
        raise NotImplementedError

    def add(self, samples: List[Sample]):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class QueueMemory(AbstractMemory):

    def __init__(self, maxlen: int):
        super(QueueMemory, self).__init__(maxlen)
        self.queue = deque(maxlen=maxlen)

    def sample(self, batch_size: int) -> List[Sample]:
        return rand.sample(self.queue, k=batch_size)

    def add(self, samples: List[Sample]):
        for sample in samples:
            self.queue.append(sample)

    def __str__(self):
        return "\n" \
               "Queue Memory.\n" \
               "Maximum capacity: " + str(self.queue.maxlen) + "\n\n"


class PrioritizedExperienceReplay(AbstractMemory):

    # ToDo: Dorobić k-step!!!

    def __init__(self, maxlen: int, model: Model, gamma: float, alpha: float = 0.5, key_scaling: int = 1):
        super(PrioritizedExperienceReplay, self).__init__(maxlen)
        self.model = model
        self.sumtree = SumTree(capacity=maxlen)
        self.key_scaling = key_scaling
        self.gamma = gamma
        self.alpha = alpha

    def sample(self, batch_size: int) -> List[Sample]:
        return [self.sumtree.sample() for _ in range(batch_size)]

    def _loss_calculate(self, sample: Sample) -> float:
        predicted_value = self.model.predict(np.array([sample.state]))[0]
        predicted_value = predicted_value[sample.action]

        q_value = sample.reward
        if sample.next_state is not None:
            q_value += self.gamma*np.max(self.model.predict(np.array([sample.next_state])))

        return (predicted_value - q_value)**2

    def add(self, samples: List[Sample]):
        for sample in samples:
            key = self._loss_calculate(sample)
            key **= self.alpha
            key = min(int(key*self.key_scaling), 1)
            self.sumtree.add(key=key, item=sample)

    def update_model(self, model: Model):
        self.model = model

    def __str__(self):
        return "\n" \
               "Prioritized Experience Replay.\n"\
               "Maximum capacity: " + str(self.sumtree.capacity) + "\n" \
               "Scaling keys parameter: " + str(self.key_scaling) + "\n"\
               "Alpha parameter (importance sampling scaling): " + str(self.gamma) + "\n"
