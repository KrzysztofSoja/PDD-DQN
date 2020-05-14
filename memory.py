import random as rand

from sample import Sample
from typing import List
from collections import deque

from keras import Model
from bintrees import FastAVLTree


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

    def __init__(self, maxlen: int, model: Model):
        super(PrioritizedExperienceReplay, self).__init__(maxlen)
        self.model = model
        self.tree = FastAVLTree()

    def sample(self, batch_size: int) -> List[Sample]:
        pass

    def add(self, samples: List[Sample]):
        pass

