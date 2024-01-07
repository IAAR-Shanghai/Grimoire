import json

import requests
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from core.llm.base import BaseLLM


class Baichuan2_53B_Chat(BaseLLM):
    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(5), reraise=True)
    def _request(self, query) -> str:
        import time
        url = self.conf['url']
        api_key = self.conf['api_key']
        secret_key = self.conf['secret_key']
        time_stamp = int(time.time())

        json_data = json.dumps({
            "model": "Baichuan2-53B",
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "parameters": {
                "temperature": self.params['temperature'],
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })

        def _calculate_md5(input_string):
            import hashlib
            md5 = hashlib.md5()
            md5.update(input_string.encode('utf-8'))
            encrypted = md5.hexdigest()
            return encrypted
        signature = _calculate_md5(secret_key + json_data + str(time_stamp))

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "X-BC-Timestamp": str(time_stamp),
            "X-BC-Signature": signature,
            "X-BC-Sign-Algo": "MD5",
        }
        res = requests.post(url, data=json_data, headers=headers)
        res = res.json()['data']['messages'][0]['content']
        return res


class GPT(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=0.01, max_new_tokens=64, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(5), reraise=True)
    def _request(self, query: str) -> str:
        client = OpenAI(api_key=self.conf['api_key'])
        res = client.chat.completions.create(
            model=self.params['model_name'],
            messages=[{"role": "user", "content": query}],
            temperature=self.params['temperature'],
            max_tokens=self.params['max_new_tokens'],
            top_p=self.params['top_p'],
        )
        real_res = res.choices[0].message.content

        token_consumed = res.usage.total_tokens
        logger.info(
            f'GPT token consumed: {token_consumed}') if self.report else ()
        return real_res
