import datetime
from typing import List

from loguru import logger

from core.data.sampler.base import BaseSampler
from core.evaluator.base import BaseEvaluator


class FewShotEvaluator(BaseEvaluator):
    def post_init(self, few_shot_sampler: BaseSampler) -> None:
        self.few_shot_sampler = few_shot_sampler

    def evaluator_info(self) -> dict:
        return {
            'setting': self.setting_name,
            'llm': self.model.params,
            'dataset': self.data_conf,
            'sampler': self.few_shot_sampler.__class__.__name__,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def scoring(self, data_point: dict) -> dict:
        try:
            few_shots = self.few_shot_sampler.draw_examples(data_point)
            result = self.model.classify(self.data_conf, data_point, few_shots)
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
