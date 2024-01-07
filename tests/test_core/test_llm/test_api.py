# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import unittest

from core.llm.api import (
    Baichuan2_53B_Chat,
    GPT,
)


class TestBaichuan253BChat(unittest.TestCase):
    def setUp(self):
        self.model = Baichuan2_53B_Chat()

    def test_request(self):
        query = "How are you?"
        response = self.model._request(query)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


class TestGPT(unittest.TestCase):
    def setUp(self):
        self.gpt35 = GPT(model_name='gpt-3.5-turbo', temperature=0.1)
        self.gpt4_0613 = GPT(model_name='gpt-4-0613', temperature=0.1)
        self.gpt4_1106 = GPT(model_name='gpt-4-1106-preview', temperature=0.1)

    def _test_request(self, model):
        query = "How are you?"
        response = model._request(query)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_request(self):
        for model in [self.gpt35, self.gpt4_0613, self.gpt4_1106]:
            with self.subTest(model=model):
                self._test_request(model)


if __name__ == '__main__':
    unittest.main()
