import json
import os
import pickle
from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import util


MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
BASE = os.path.dirname(os.path.abspath(__file__))


def compute_similarity_ranks(test_ebds: np.ndarray, train_ebds: np.ndarray, top_k: int = 1000) -> List[List[int]]:
    results = util.semantic_search(test_ebds, train_ebds, top_k=top_k)
    sim_ranks = [
        [item['corpus_id'] for item in result]
        for result in results
    ]
    return sim_ranks


if __name__ == '__main__':
    for data_name in os.listdir(BASE):
        if not os.path.isdir(os.path.join(BASE, data_name)):
            continue

        train_path = os.path.join(
            BASE, data_name, f'ebd_train_{MODEL_NAME.split("/")[-1]}.pickle')
        with open(train_path, 'rb') as f:
            train_ebds = pickle.load(f)

        test_path = os.path.join(
            BASE, data_name, f'ebd_test_{MODEL_NAME.split("/")[-1]}.pickle')
        with open(test_path, 'rb') as f:
            test_ebds = pickle.load(f)

        test_data_path = os.path.join(BASE, data_name, f'test.json')
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)

        list_ranks = compute_similarity_ranks(
            test_ebds, train_ebds, top_k=2000)
        dict_ranks = {test_data[i]['text']: rank for i,
                      rank in enumerate(list_ranks)}
        ranks_path = os.path.join(
            BASE, data_name, f'sims_{MODEL_NAME.split("/")[-1]}.pickle')
        with open(ranks_path, 'wb') as f:
            pickle.dump(dict_ranks, f)
        logger.info(f'Similarity ranks of {data_name} saved.')
