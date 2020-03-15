import random as rand

from sample import Sample
from typing import List
from collections import deque


class AbstractMemory:

    def sample(self, batch_size: int) -> List[Sample]:
        raise NotImplementedError

    def add(self, samples: List[Sample]):
        raise NotImplementedError


class QueueMemory(AbstractMemory):

    def __init__(self, size: int):
        super(QueueMemory, self).__init__()
        self.queue = deque(maxlen=size)

    def sample(self, batch_size: int) -> List[Sample]:
        return rand.sample(self.queue, k=batch_size)

    def add(self, samples: List[Sample]):
        for sample in samples:
            self.queue.append(sample)
