import datetime
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from core.evaluator.base import BaseEvaluator
from core.llm.base import BaseLLM


class GrimoireRankEvaluator(BaseEvaluator):
    def post_init(
            self,
            embedding_model_name: str,
            grimoire_generator: BaseLLM,
            grimoire_dir: str,
            filter_out_contains: List[str] = [],
    ) -> None:
        self.grimoires = self._read_all_grimoires(
            grimoire_dir,
            'grimoire_' + self.data_conf['data_name'] + '_' + grimoire_generator.params['model_name']
        )

        # Conduct filtering
        for grimoire_name in list(self.grimoires.keys()):
            for word in filter_out_contains:
                if word in grimoire_name:
                    del self.grimoires[grimoire_name]
                    break
            if '-shot-hard' in grimoire_name:
                if '-shot-hard-for-' + self.model.params['model_name'] in grimoire_name:
                    continue
                else:
                    del self.grimoires[grimoire_name]

        self.embedding_model_name = embedding_model_name
        self.ebd_model = SentenceTransformer(embedding_model_name)
        self.grimoire_ebds = {grimoire_name: self.ebd_model.encode(grimoire) for grimoire_name, grimoire in self.grimoires.items()}
        logger.info('Grimoire embeddings are ready.')

    def evaluator_info(self) -> dict:
        return {
            'setting': self.setting_name,
            'llm': self.model.params,
            'dataset': self.data_conf,
            'grimoires': self.grimoires,
            'embedding_model': self.embedding_model_name,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def scoring(self, data_point: dict) -> dict:
        try:
            grimoire_name, sim_score = self._get_most_similar_grimoire(self.grimoire_ebds, data_point, self.ebd_model)
            grimoire = self.grimoires[grimoire_name]
            result = self.model.classify(self.data_conf, data_point, grimoire=grimoire)
        except Exception as e:
            result = ''
            logger.warning(repr(e))
        return {
            'correct': result.lower() == data_point['ans_text'].lower(),
            'output': result,
            'grimoire_name': grimoire_name,
            'simlarity': sim_score,
            'valid': result.lower() in [label.lower() for label in self.data_conf['data_label_list']] \
                     or result.lower() in data_point['text'],
        }

    def batch_scoring(self, dataset: List[dict]) -> List[dict]:
        results = []
        for data_point in tqdm(dataset, desc=self.model.params['model_name']):
            result = self.scoring(data_point)
            results.append({**result, 'original_data': data_point})
        return results

    def compute_overall(self, valid_results: List[dict]) -> dict:
        return {
            'accuracy': sum([result['correct'] for result in valid_results]) / len(valid_results),
            'valid_num': len(valid_results),
            'grimoire_usage': dict(Counter([result['grimoire_name'] for result in valid_results])),
        }

    @staticmethod
    def _read_all_grimoires(grimoire_dir: str, grimoire_filename_start: str) -> Dict[str, str]:
        """Get all grimoires from grimoire_dir.
        
        Return:
            dict: {grimoire_name: grimoire}
        """
        grimoires = {}
        for filename in os.listdir(grimoire_dir):
            if not filename.startswith(grimoire_filename_start):
                continue
            grimoire_path = os.path.join(grimoire_dir, filename)
            with open(grimoire_path, 'r') as f:
                grimoire_dict = json.load(f)
            grimoires['profound-' + filename] = grimoire_dict['profound_grimoire']
            grimoires['simple-' + filename] = grimoire_dict['simple_grimoire']
        return grimoires

    @staticmethod
    def _get_most_similar_grimoire(grimoire_ebds: Dict[str, Any], data_point: dict, ebd_model: SentenceTransformer) -> Tuple[str, float]:
        """Get the most similar grimoire for a data point.

        Return:
            tuple: (grimoire_name, similarity_score)
        """

        data_point_ebd = ebd_model.encode(data_point['text'])
        target_idx, sim_score = util.semantic_search(data_point_ebd, np.array(list(grimoire_ebds.values())), top_k=1)[0][0].values()
        grimoire_name = list(grimoire_ebds.keys())[target_idx]
        return grimoire_name, sim_score
