from sample import Sample
from typing import List


class AbstractMemory:

    def __init__(self):
        raise NotImplementedError

    def sample(self, batch_size: int) -> List[Sample]:
        raise NotImplementedError

    def add(self, samples: List[Sample]):
        raise NotImplementedError
