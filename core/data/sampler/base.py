from abc import ABC, abstractmethod
from typing import List


class BaseSampler(ABC):
    def __init__(self, data: List[dict], cnt: int = 8):
        self.data = data
        self.cnt = cnt

        if cnt > len(data):
            raise ValueError('cnt must be less than or equal to the length of data')

    @abstractmethod
    def draw_examples(self, for_data_point: dict) -> List[dict]:
        return self.data[:self.cnt]
