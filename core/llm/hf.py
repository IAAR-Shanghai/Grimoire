import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from core.llm.base import BaseLLM


class AquilaChat2_7B(BaseLLM):
    # For a more formal inference approach, please refer to https://huggingface.co/BAAI/AquilaChat2-7B.
    def _request(self, query: str) -> str:
        local_path = self.conf['url']
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(input_ids, **gen_kwargs)[0]
        response = tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response