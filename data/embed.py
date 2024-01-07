import json
import os
from typing import List

from sentence_transformers import SentenceTransformer


MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
BASE = os.path.dirname(os.path.abspath(__file__))


def embed(model, data: List[dict], path: str) -> None:
    ebds = model.encode([item['text'] for item in data])
    with open(path, 'wb') as f:
        f.write(ebds.dumps())


if __name__ == '__main__':
    print('Load model:', MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    for data_name in os.listdir(BASE):
        if not os.path.isdir(os.path.join(BASE, data_name)):
            continue
        print('Embed:', data_name)

        for data_type in ['train', 'test']:
            path = os.path.join(BASE, data_name, f'{data_type}.json')
            with open(path, 'r') as f:
                data = json.load(f)
            ebd_path = os.path.join(
                BASE, data_name, f'ebd_{data_type}_{MODEL_NAME.split("/")[-1]}.pickle')
            embed(model, data, ebd_path)
