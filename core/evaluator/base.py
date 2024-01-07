import copy
import datetime
import json
import os
from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import List

from loguru import logger
from tqdm import tqdm

from core.llm.base import BaseLLM

OUTPUT_DIR = './outputs'


class BaseEvaluator(ABC):
    def __init__(
            self,
            model: BaseLLM,
            data_conf: dict,
            dataset: List[dict],
            setting_name: str,
            output_dir: str = OUTPUT_DIR,
            process_num: int = 1,
    ):
        """
        Args:
            model (BaseLLM): The large language model to be evaluated.
            dataset (list[dict]): The dataset for evaluation.
            output_dir (str): The directory for result output and caching.
        """
        self.model = copy.deepcopy(model)
        self.data_conf = data_conf
        self.dataset = dataset

        self.setting_name = setting_name
        data_name = data_conf['data_name']
        model_name = model.params['model_name']
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'{setting_name}_{data_name}_{model_name}_{timestamp}.json'
        self.output_path = os.path.join(output_dir, output_name)

        self.process_num = process_num

    @abstractmethod
    def post_init(self):
        # self.input1 = input1
        # self.input2 = input2
        ...

    @abstractmethod
    def evaluator_info(self) -> dict:
        return {
            'setting': self.setting_name,
            'llm': self.model.params,
            'dataset': self.data_conf,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            # 'key_1': Value,
            # 'key_2': Value,
            # ...
        }

    @abstractmethod
    def scoring(self, data_point: dict) -> dict:
        """Invoke self.model to evaluate data_point.

        Returns:
            dict: A result dictionary containing three mandatory fields: `metrics`, `log`, `valid`.

        Note:
            Make sure that no exception will be raised in this method.
        """
        return {
            'metrics': {
                # Numerical results to be recorded by subclasses, mandatory.
                # Such as accuracy, recall, bleu, rouge, etc.
            },
            'log': {
                # String results to be recorded by subclasses, optional.
                # Such as model output.
            },
            'valid': ...
                # Boolean result to be recorded by subclasses, indicating whether the evaluation is valid, mandatory.
                # True or False.
        }
    
    def batch_scoring(self, dataset: List[dict]) -> List[dict]:
        """Perform batch scoring on the given dataset.
        
        Args:
            dataset (list[dict]): The dataset for evaluation.
        
        Returns:
            list[dict]: List of results.
        """

        with Pool(self.process_num) as pool:
            results = list(tqdm(
                pool.imap(self.scoring, dataset),
                total = len(dataset),
                desc = self.model.params['model_name']
            ))
            # results = pool.map(self.scoring, dataset)


        results = [
            {**result, 'original_data': data_point}
            for result, data_point
            in zip(results, dataset)
        ]

        return results
    
    @abstractmethod
    def compute_overall(self, valid_results: List[dict]) -> dict:
        """Extract and aggregate results from individual data points in the results.
        For example, calculate mean, variance, etc.

        Returns:
            dict: A result dictionary that can store any number and form of fields.
        """
        return {
            # 'Metric1': Value,
            # 'Metric2': Value,
            # ...
        }

    def save_output(self, output: dict) -> None:
        """Save evaluation results."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def run(self) -> dict:
        """Run a complete evaluation.

        Returns:
            dict: Output dictionary contains fields such as: info, overall, results, etc.
        """
        info = self.evaluator_info()

        results = self.batch_scoring(self.dataset)
        valid_results = [result for result in results if result['valid']]

        try:
            overall = self.compute_overall(valid_results) if len(valid_results) > 0 else {}
        except Exception as e:
            logger.warning(repr(e))
            overall = dict()

        self.save_output(output:={'info': info, 'overall': overall, 'results': results})
        print(f'Output saved at {self.output_path}!')
        return output
