# @Author : Shichao Song
# @Email  : song.shichao@outlook.com

"""Unit tests for the core.llm.remote module.

This module contains unittests for the llm deployed remotely.

Note:
    These tests perform real requests to external APIs. Be cautious of network availability,
    API rate limits, and potential costs associated with making real requests during testing.
"""


import unittest

from core.llm.remote import (
    GPT_transit,
)


class TestGPTTransit(unittest.TestCase):
    def setUp(self):
        self.gpt35 = GPT_transit(model_name='gpt-3.5-turbo')
        self.gpt4_0613 = GPT_transit(model_name='gpt-4-0613')
        self.gpt4_1106 = GPT_transit(model_name='gpt-4-1106-preview')

    def _test_request(self, model):
        query = "How are you?"
        response = model._request(query)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print(response)

    def test_request(self):
        for model in [self.gpt35, self.gpt4_0613, self.gpt4_1106]:
            with self.subTest(model=model):
                self._test_request(model)


if __name__ == '__main__':
    unittest.main()
