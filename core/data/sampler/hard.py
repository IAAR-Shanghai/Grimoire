import json
import os
from multiprocessing import Pool
from typing import List, Tuple

from loguru import logger
from tqdm import tqdm

from core.data.sampler.base import BaseSampler
from core.data.sampler.utils import cluster_by_label, draw_from_clusters
from core.llm.base import BaseLLM


CACHE_KEEP_HARD_DIR = './.cache/keep_hard'


class KeepHardSampler(BaseSampler):
    def __init__(
        self,
        data: List[dict],
        data_conf: dict,
        for_model: BaseLLM,
        cnt: int = 8,
        hard_ratio: float = 0.5,
        cluster: bool = True,
        process_num: int = 10
    ):
        hard_cnt = int(hard_ratio * cnt)
        easy_cnt = cnt - hard_cnt

        self.for_model = for_model
        self.data_conf = data_conf
        data_name = data_conf['data_name']

        # ─── Cache Or Read ────────────────────────────────────────────

        filename = f'keep_hard_{data_name}_{for_model.params["model_name"]}.json'
        path = os.path.join(CACHE_KEEP_HARD_DIR, filename)
        if not os.path.exists(path):
            logger.info(f'Labeling easy and hard data in {data_name} for {for_model.params["model_name"]}')
            easy, hard = self._label_easy_and_hard(data, process_num)
            self._save_easy_and_hard(easy, hard, path)
        else:
            easy, hard = self._read_esay_and_hard(path)

        # ─── Cluster Processing ───────────────────────────────────────

        if cluster:
            easy_clusters = cluster_by_label(easy)
            hard_clusters = cluster_by_label(hard)
        else:
            easy_clusters = {'easy': easy}
            hard_clusters = {'hard': hard}

        self.examples = draw_from_clusters(easy_clusters, easy_cnt) \
            + draw_from_clusters(hard_clusters, hard_cnt)

    def draw_examples(self, for_data_point: dict = None) -> List[dict]:
        return self.examples

    def _label_easy_and_hard(
        self,
        data: List[dict],
        process_num: int,
    ) -> Tuple[list, list]:
        """Label easy and hard data."""

        with Pool(process_num) as pool:
            corrects = list(tqdm(
                pool.imap(self._classify, data),
                total = len(data),
                desc = self.for_model.params['model_name']
            ))

        easy = [data_point for data_point, correct in zip(data, corrects) if correct]
        hard = [data_point for data_point, correct in zip(data, corrects) if not correct]
        return easy, hard

    def _classify(self, data_point: dict) -> bool:
        result = self.for_model.classify(self.data_conf, data_point)
        correct = result.lower() == data_point['ans_text'].lower()
        return correct

    @staticmethod
    def _save_easy_and_hard(easy: List[dict], hard: List[dict], path: str) -> None:
        with open(path, 'w') as f:
            json.dump({
                'easy': easy,
                'hard': hard,
            }, f, indent=4)

    @staticmethod
    def _read_esay_and_hard(path: str) -> None:
        with open(path, 'r') as f:
            data = json.load(f)
        return data['easy'], data['hard']
