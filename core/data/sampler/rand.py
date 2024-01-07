import random
from collections import defaultdict
from typing import Dict, List

from core.data.sampler.base import BaseSampler
from core.data.sampler.utils import cluster_by_label, draw_from_clusters


class RandomSampler(BaseSampler):
    def __init__(
        self,
        data: List[dict],
        cnt: int = 8,
        cluster: bool = True,
        identical: bool = False,
        seed: int = None,
    ):
        self.data = data
        self.cnt = cnt
        self.cluster = cluster
        self.identical = identical
        self.seed = seed

        if cnt > len(data):
            raise ValueError('cnt must be less than or equal to the length of data')

        random.seed(seed) if seed is not None else ...
        random.shuffle(self.data)

        if cluster:
            self.clusters = cluster_by_label(self.data)
        else:
            self.clusters = {0: self.data}
        
        self.identical_examples = draw_from_clusters(self.clusters, self.cnt)

    def draw_examples(self, for_data_point: dict = None) -> List[dict]:
        if self.identical:
            examples = self.identical_examples
        else:
            seed = self._gen_seed(for_data_point)
            examples = draw_from_clusters(self.clusters, self.cnt, seed=seed)
        return examples

    @staticmethod
    def _gen_seed(data_point: dict) -> int:
        hash_value = hash(str(data_point))
        seed = hash_value % (2**16)
        return seed
