import pickle
import random
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def read_embeddings(embeddings_path: str) -> np.ndarray:
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return dot_product / (norm_u * norm_v)


def get_the_center(indices: np.ndarray, embeddings: np.ndarray) -> int:
    """Get the index of the embedding that is closest to the center of embeddings in embeddings[indices]."""
    if type(indices) == list:
        indices = np.array(indices)
    target_ebds = embeddings[indices]
    center = np.array(target_ebds).mean(axis=0)
    similarities = np.array([cosine_similarity(x, center) for x in target_ebds])
    return indices[np.argmax(similarities)]


def cluster_by_label(data: List[dict]) -> Dict[Any, List[dict]]:
    """Cluster data by label."""
    clusters = defaultdict(list)
    for data_point in data:
        clusters[data_point['ans_text']].append(data_point)
    return clusters


def cluster_by_label_indices(data: List[dict]) -> Dict[Any, List[int]]:
    """Cluster data by label. Return the indices of data points in each cluster."""
    clusters = defaultdict(list)
    for i, data_point in enumerate(data):
        clusters[data_point['ans_text']].append(i)
    return clusters


def draw_from_clusters(clusters: Dict[Any, List[dict]], cnt: int, seed: int = None) -> List[dict]:
    """Draw `cnt` data points from clusters with the same number of data points for each cluster.
    
    Args:
        clusters: A dict of clusters.
        cnt: The number of data points to draw.
        seed: The seed for random.
    
    Notes:
        If seed is None, the data points are drawn sequentially from the clusters;
        otherwise, the data points are drawn randomly from the clusters.
    
    Returns:
        A list of data points.
    """
    random.seed(seed) if seed is not None else ...

    if cnt > sum([len(x) for x in clusters.values()]):
        raise ValueError('cnt must be less than or equal to the length of data')

    list_clusters  = sorted(list(clusters.values()), key=lambda x: len(x))
    num_clusters = len(list_clusters)

    cnt_per_cluster = cnt // num_clusters
    remainder = cnt % num_clusters

    cnts = [cnt_per_cluster] * num_clusters
    for i in range(remainder):
        cnts[i] += 1

    examples = []
    remained_sample_cnt = 0
    for i in range(num_clusters):
        target_sample_cnt = cnts[i] + remained_sample_cnt
        if target_sample_cnt == 0:
            break
        real_sample_cnt = min(target_sample_cnt, len(list_clusters[i]))
        if seed is None:
            examples += list_clusters[i][:real_sample_cnt]
        else:
            examples.extend(random.sample(list_clusters[i], real_sample_cnt))
        remained_sample_cnt = target_sample_cnt - real_sample_cnt

    return examples
