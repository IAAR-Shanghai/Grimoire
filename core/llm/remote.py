import json

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

from core.llm.base import BaseLLM


TIMEOUT = 45  # Unified timeout for requesting in remote.py


class GPT_transit(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=0.01, max_new_tokens=64, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(5), reraise=True)
    def _request(self, query: str) -> str:
        url = self.conf['url']
        payload = json.dumps({
            "model": self.params['model_name'],
            "messages": [{"role": "user", "content": query}],
            "temperature": self.params['temperature'],
            'max_tokens': self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
        })
        headers = {
            'token': self.conf['token'],
            'User-Agent': self.conf['user_agent'],
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': self.conf['host'],
            'Connection': 'keep-alive'
        }
        res = requests.request("POST", url, headers=headers,
                               data=payload, timeout=TIMEOUT)
        res = res.json()
        real_res = res["choices"][0]["message"]["content"]
        return real_res
