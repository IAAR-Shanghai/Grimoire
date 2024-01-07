import copy
import os
from abc import ABC, abstractmethod
from typing import List

from loguru import logger

from core.load_conf import load_yaml_conf


PROMPT_DIR = './prompts'
LLM_CONF_PATH = './configs/llm.yaml'


class BaseLLM(ABC):
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.01,
        max_new_tokens: int = 64,
        top_p: float = 0.9,
        top_k: int = 5,
        **more_params
    ):
        self.params = {
            'model_name': model_name if model_name else self.__class__.__name__,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'top_k': top_k,
            **more_params
        }
        # Load secret token / id / key / url of current LLM
        self.conf = load_yaml_conf(LLM_CONF_PATH)[self.__class__.__name__]

    @abstractmethod
    def _request(self, query: str) -> str:
        """Without further processing the response of the request; simply return the string."""
        return ''

    @staticmethod
    def _read_prompt_template(filename: str) -> str:
        path = os.path.join(PROMPT_DIR, filename)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''

    def update_params(self, inplace: bool = True, **params):
        """Update parameters either in-place or create a new object with updated parameters.

        Args:
            inplace (bool, optional): If True, update parameters in-place. If False, create a new object with updated parameters. Default is True.
            **params: Keyword arguments representing parameters to be updated.

        Returns:
            BaseLLM: An instance of the class with updated parameters.

        Examples:
            Update parameters in-place:
            ```
            >>> obj.update_params(param1=20)
            >>> print(obj.params)
            {'param1': 20}
            ```

            Create a new object with updated parameters:
            ```
            >>> new_obj = obj.update_params(False, param1=20)
            >>> print(obj.params)
            {'param1': 10}
            >>> print(new_obj.params)
            {'param1': 20}
            ```
        """
        if inplace:
            self.params.update(params)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.params.update(params)
            return new_obj

    def safe_request(self, query: str) -> str:
        """Safely make a request to the language model, handling exceptions."""
        try:
            response = self._request(query)
        except Exception as e:
            logger.warning(repr(e))
            response = ''
        return response

    # ─── Prompt Engineering ───────────────────────────────────────────────

    def classify(self, data_conf: dict, test_data: dict, few_shot_data: List[dict] = None, grimoire: str = None):
        """Classify the test data based on the specified task configuration (few-shot / zero-shot / with-grimoire).

        Args:
            data_conf (dict): A dictionary containing task configuration parameters, including 'task_type' and 'task_description'.
            test_data (dict): Test data dictionary with 'text' key representing the real question.
            few_shot_data (List[dict], optional): Few-shot examples used for the classification. Each dictionary in the list should contain 'text' and 'ans_text' keys.
            grimoire (str, optional): A string representing additional information for with-grimoire setting.

        Returns:
            str: The predicted label / result.

        Notes:
            - If 'few_shot_data' is provided, the function operates in the few-shot setting.
            - If 'few_shot_data' is not provided but 'grimoire' is provided, the function operates in the with-grimoire setting.
            - If both 'few_shot_data' and 'grimoire' are not provided, the function operates in the zero-shot setting.

        Raises:
            Any exceptions raised during the API request or processing.

        Example:
            ```
            >>> llm = AnImplementedLLMClass()
            >>> data_config = {'task_type': 'classification', 'task_description': 'Categorize text'}
            >>> test_data = {'text': 'Sample text for classification'}
            >>> few_shot_data = [{'text': 'Example 1', 'ans_text': 'Category A'}, {'text': 'Example 2', 'ans_text': 'Category B'}]
            >>> result = llm.classify(data_config, test_data, few_shot_data)
            ```
        """

        # ─── Construct The Few-shot Examples Or The Grimire ───────────

        if few_shot_data is not None and len(few_shot_data) > 0:  # few-shot setting
            examples_str = 'Here are some examples:\n\n' + '\n\n'.join([
                shot['text'] + '\nAnswer: ' + shot['ans_text']  # Few-shot examples
                for shot in few_shot_data
            ])
            examples_or_grimoire = examples_str
        else:  # zero-shot setting or with-grimoire setting
            grimoire_str = grimoire if grimoire is not None else ''
            examples_or_grimoire = grimoire_str

        # ─── Construct The Prompt From The Template ───────────────────

        prompt_template = self._read_prompt_template('classify.txt')
        prompt = prompt_template.format(
            task_type = data_conf['task_type'],
            task_description = data_conf['task_description'],
            examples_or_grimoire = examples_or_grimoire,
            question = test_data['text'] + '\nAnswer: ',
        )

        # ─── Query And Get The Returned Label ─────────────────────────

        res = self.safe_request(prompt)
        real_res = res.split('Answer: ')[-1].strip().replace('.', '').replace(',', '').lower()
        return real_res
    
    def generate_grimoire(self, data_conf: dict, data: List[dict]):
        examples_str = 'Examples:\n\n' + '\n\n'.join([
            data_point['text'] + '\nAnswer: ' + data_point['ans_text']
            for data_point in data
        ])
        if len(data) == 0:
            examples_str = ''

        deluxe_prompt_template = self._read_prompt_template('generate_grimoire_deluxe.txt')
        basic_prompt_template = self._read_prompt_template('generate_grimoire_basic.txt')

        deluxe_prompt = deluxe_prompt_template.format(
            task_description=data_conf['task_description'],
            examples=examples_str,
        )
        deluxe_grimoire = self.safe_request(deluxe_prompt)

        basic_prompt = basic_prompt_template.format(
            deluxe_grimoire=deluxe_grimoire
        )
        basic_grimoire = self.safe_request(basic_prompt)

        start = '\nBelow are some skills needed to solve the task; you need to carefully learn and consider the process and methods step by step:\n\n'
        deluxe_grimoire = start + deluxe_grimoire
        basic_grimoire = start + basic_grimoire

        grimoire = {'deluxe_grimoire': deluxe_grimoire,
                    'basic_grimoire': basic_grimoire}

        return grimoire
