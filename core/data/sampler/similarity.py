import pickle
from typing import List

from core.data.sampler.base import BaseSampler
from core.data.sampler.utils import cluster_by_label, draw_from_clusters


class SimilaritySampler(BaseSampler):
    def __init__(
        self,
        data: List[dict],
        similarity_ranks_path: str,
        cnt: int = 8,
        cluster: bool = True,
        reverse: bool = False,
    ):
        self.data = data
        self.sim_ranks = self.read_sim_ranks(similarity_ranks_path)
        self.cnt = cnt
        self.cluster = cluster
        self.reverse = reverse

    def draw_examples(self, for_data_point: dict = None) -> List[dict]:
        the_text = for_data_point['text']
        indices = self.sim_ranks[the_text]
        data_points = [self.data[i] for i in indices]
        if self.cluster:
            clusters = cluster_by_label(data_points)
        else:
            clusters = {0: data_points}
        examples = draw_from_clusters(clusters, self.cnt)
        return examples[::-1] if self.reverse else examples

    @staticmethod
    def read_sim_ranks(similarity_ranks_path: str) -> dict:
        with open(similarity_ranks_path, 'rb') as f:
            sim_ranks = pickle.load(f)
        return sim_ranks
