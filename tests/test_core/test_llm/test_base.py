# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import unittest
from unittest.mock import MagicMock, patch

from core.llm.base import BaseLLM


class ConcreteLLM(BaseLLM):
    """An alternative to BaseLLM since BaseLLM is an abstract class and cannot be instantiated directly."""
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
        self.conf = {'key': 'value'}

    def _request(self, query: str) -> str:
        return 'Mocked Response'


class TestBaseLLM(unittest.TestCase):

    def setUp(self):
        # Initialize an instance of BaseLLM's child class for testing
        self.llm_instance = ConcreteLLM()

    def test_init(self):
        self.assertEqual(self.llm_instance.params['model_name'], 'ConcreteLLM')
        self.assertEqual(self.llm_instance.params['temperature'], 0.01)
        self.assertEqual(self.llm_instance.params['max_new_tokens'], 64)
        self.assertEqual(self.llm_instance.params['top_p'], 0.9)
        self.assertEqual(self.llm_instance.params['top_k'], 5)

    def test_update_params(self):
        new_params = {'model_name': 'NewModel', 'temperature': 0.8}
        updated_instance = self.llm_instance.update_params(**new_params)
        self.assertEqual(updated_instance.params['model_name'], 'NewModel')
        self.assertEqual(updated_instance.params['temperature'], 0.8)

    @patch.object(ConcreteLLM, '_request', return_value='mocked_response')
    def test_safe_request(self, mock_request):
        response = self.llm_instance.safe_request('query')
        mock_request.assert_called_once_with('query')
        self.assertEqual(response, 'mocked_response')

    def test_read_prompt_template(self):
        with patch('builtins.open', return_value=open('./prompts/classify.txt', 'r')):
            template = self.llm_instance._read_prompt_template('classify.txt')
        self.assertTrue(template.startswith('You are an expert in'))


class TestClassifyFunction(unittest.TestCase):
    def setUp(self):
        self.llm = ConcreteLLM()

    def test_classify_zero_shot(self):
        self.llm._request = MagicMock(return_value='ans: Category A')
        data_conf = {'task_type': 'classification', 'task_description': 'Categorize text'}
        test_data = {'text': 'Sample text for classification'}
        result = self.llm.classify(data_conf, test_data)
        self.llm._request.assert_called_once()
        self.assertEqual(result, 'Category A')

    def test_classify_few_shot(self):
        self.llm._request = MagicMock(return_value='ans: Category B')
        data_conf = {'task_type': 'classification', 'task_description': 'Categorize text'}
        test_data = {'text': 'Sample text for classification'}
        few_shot_data = [{'text': 'Example 1', 'ans_text': 'Category A'}, {'text': 'Example 2', 'ans_text': 'Category B'}]
        result = self.llm.classify(data_conf, test_data, few_shot_data)
        self.llm._request.assert_called_once()
        self.assertEqual(result, 'Category B')

    def test_classify_with_grimoire(self):
        self.llm._request = MagicMock(return_value='ans: Category C')
        data_conf = {'task_type': 'classification', 'task_description': 'Categorize text'}
        test_data = {'text': 'Sample text for classification'}
        grimoire = 'Additional information for with-grimoire setting.'
        result = self.llm.classify(data_conf, test_data, grimoire=grimoire)
        self.llm._request.assert_called_once()
        self.assertEqual(result, 'Category C')


if __name__ == '__main__':
    unittest.main()
