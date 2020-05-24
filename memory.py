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


class QueueMemory(AbstractMemory):

    def __init__(self, maxlen: int):
        super(QueueMemory, self).__init__(maxlen)
        self.queue = deque(maxlen=maxlen)

    def sample(self, batch_size: int) -> List[Sample]:
        return rand.sample(self.queue, k=batch_size)

    def add(self, samples: List[Sample]):
        for sample in samples:
            self.queue.append(sample)


class PrioritizedExperienceReplay(AbstractMemory):

    # ToDo: Dorobić k-step!!!
    # ToDo: Czy warto implementować tu Double DQN

    def __init__(self, maxlen: int, model: Model, key_scaling: int, gamma: float):
        super(PrioritizedExperienceReplay, self).__init__(maxlen)
        self.model = model
        self.sumtree = SumTree(capacity=maxlen)
        self.key_scaling = key_scaling
        self.gamma = gamma

    def sample(self, batch_size: int) -> List[Sample]:
        return [self.sumtree.sample() for _ in range(batch_size)]

    def _loss_calculate(self, sample: Sample) -> float:
        predicted_value = self.model.predict(np.array([sample.state]))[0]
        predicted_value = predicted_value[sample.action]

        q_value = sample.reward
        if sample.next_state is not None:
            q_value += self.gamma*np.max(self.model.predict(np.array([sample.next_state])))

        return (predicted_value - q_value)**2

    def add(self, samples: List[Sample]): #ToDo: Możeby nie dodawać elementów mniejszych na epsylon
        for sample in samples:
            key = self._loss_calculate(sample)
            key = int(key*self.key_scaling)
            self.sumtree.add(key=key, item=sample)

    def update_model(self, model: Model):
        self.model = model

