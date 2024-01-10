import datetime
import json
import os
from typing import List

from loguru import logger

from core.data.sampler.base import BaseSampler
from core.evaluator.base import BaseEvaluator
from core.llm.base import BaseLLM


class WithGrimoireEvaluator(BaseEvaluator):
    def post_init(
            self,
            few_shot_sampler: BaseSampler,
            verbose: bool,
            grimoire_generator: BaseLLM,
            grimoire_dir: str,
    ) -> None:
        self.few_shot_sampler = few_shot_sampler
        self.verbose = verbose

        self.grimoire_generator = grimoire_generator
        self.grimoire_dir = grimoire_dir
        self.grimoire_filename = 'grimoire_' + \
            self.data_conf['data_name'] + '_' + \
            self.grimoire_generator.params['model_name'] + '_' + \
            self.setting_name.split('-with-grimoire')[0] + '.json'
        
        if 'keep-hard' in self.setting_name:  # Special case
            self.grimoire_filename = self.grimoire_filename.replace(
                '-shot-hard',
                '-shot-hard-for-' + few_shot_sampler.for_model.params['model_name']
            )

        self.grimoire_path = os.path.join(self.grimoire_dir, self.grimoire_filename)

        if not os.path.exists(self.grimoire_path):
            few_shots = self.few_shot_sampler.draw_examples(for_data_point={})
            grimoire = self.grimoire_generator.generate_grimoire(self.data_conf, few_shots)
            self._save_grimoire(grimoire)

    def evaluator_info(self) -> dict:
        return {
            'setting': self.setting_name,
            'llm': self.model.params,
            'dataset': self.data_conf,
            'sampler': self.few_shot_sampler.__class__.__name__,
            'girmoire_generator': self.grimoire_generator.params,
            'grimoire_path': self.grimoire_path,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def scoring(self, data_point: dict) -> dict:
        try:
            grimoire = self._read_grimoire()
            result = self.model.classify(self.data_conf, data_point, grimoire=grimoire)
        except Exception as e:
            result = ''
            logger.warning(repr(e))
        return {
            'correct': result.lower() == data_point['ans_text'].lower(),
            'output': result,
            'valid': result.lower() in [label.lower() for label in self.data_conf['data_label_list']] \
                     or result.lower() in data_point['text'],
        }

    def compute_overall(self, valid_results: List[dict]) -> dict:
        return {
            'accuracy': sum([result['correct'] for result in valid_results]) / len(valid_results),
            'valid_num': len(valid_results),
        }

    def _save_grimoire(self, grimoire: str) -> None:
        """Save grimoire to a file and return the filename."""
        with open(self.grimoire_path, 'w', encoding='utf-8') as f:
            json.dump(grimoire, f, ensure_ascii=False, indent=4)

    def _read_grimoire(self) -> str:
        """Read grimoire from a file and return the content."""
        with open(self.grimoire_path, 'r', encoding='utf-8') as f:
            grimoire = json.load(f)
        return grimoire['profound_grimoire' if self.verbose else 'simple_grimoire']
