from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from core.data.sampler.base import BaseSampler
from core.data.sampler.utils import (cluster_by_label_indices, get_the_center,
                                     read_embeddings)


class HierarchicalSampler(BaseSampler):
    def __init__(
        self,
        data: List[dict],
        data_conf: dict,
        embeddings_path: str,
        num_clusters: int = 2,
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
            clusters_by_ebds = self._hierarchical_clustering(partial_ebds, num_clusters)
            centers = [get_the_center(indices, partial_ebds) for indices in clusters_by_ebds.values()]
            self.examples.extend([partial_data[idx] for idx in centers])

    def draw_examples(self, for_data_point: dict) -> List[dict]:
            return self.examples

    @staticmethod
    def _hierarchical_clustering(embeddings: np.ndarray, num_clusters: int) -> Dict[int, List]:
        num_embeddings = len(embeddings)
        cosine_similarity_matrix = np.zeros((num_embeddings, num_embeddings))

        cosine_similarity_matrix = 1 - np.dot(embeddings, embeddings.T) / np.outer(np.linalg.norm(embeddings, axis=1), np.linalg.norm(embeddings, axis=1))

        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='average')

        cluster_labels = clustering.fit_predict(cosine_similarity_matrix)

        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = [idx]
            else:
                clusters[label].append(idx)

        return clusters
