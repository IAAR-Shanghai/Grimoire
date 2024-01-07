from collections import defaultdict
from typing import Dict, List

import numpy as np
from loguru import logger

from core.data.sampler.base import BaseSampler
from core.data.sampler.utils import (cluster_by_label_indices,
                                     cosine_similarity, get_the_center,
                                     read_embeddings)


class KMeansSampler(BaseSampler):
    def __init__(
        self,
        data: List[dict],
        data_conf: dict,
        embeddings_path: str,
        num_clusters: int = 2,
        seed: int = None,
        max_iter: int = 150
    ):
        if num_clusters > len(data):
            raise ValueError('cnt must be less than or equal to the length of data')

        num_distinct_labels = len(data_conf['data_label_list'])
        if num_distinct_labels == 0:
            clusters_by_label = {0: list(range(len(data)))}
            num_clusters *= 2  # to improve fairness
        else:
            clusters_by_label = cluster_by_label_indices(data)
        
        all_embeddings = read_embeddings(embeddings_path)

        self.examples = []
        for cluster in clusters_by_label.values():
            cluster_indices = np.array(cluster)
            partial_data = np.array(data)[cluster_indices]
            partial_ebds = np.array(all_embeddings)[cluster_indices]
            clusters_by_ebd = self._k_means(partial_ebds, num_clusters, max_iter, seed)
            centers = [get_the_center(indices, partial_ebds) for indices in clusters_by_ebd.values()]
            self.examples.extend([partial_data[idx] for idx in centers])

    def draw_examples(self, for_data_point: dict) -> List[dict]:
            return self.examples

    @staticmethod
    def _k_means(embeddings: np.ndarray, k: int, max_iter: int, seed: int) -> Dict[int, List]:
        """K-means clustering algorithm. Return clusters of indices."""
        np.random.seed(seed)
        center_indices = np.random.choice(len(embeddings), k, replace=False)  # shape: 1, k
        centers = embeddings[center_indices]  # shape: k, ebd

        for iter in range(max_iter):
            similarities = np.array([[cosine_similarity(x, center) for center in centers] for x in embeddings])  # shape: len, k

            labels = np.argmax(similarities, axis=1)  # shape: len, 1
            new_centers = np.array([embeddings[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centers == new_centers):
                logger.info(f'Reach convergence at {iter}')
                break
            centers = new_centers
        
        clusters = defaultdict(list)
        [clusters[label].append(i) for i, label in enumerate(labels)]

        return clusters
