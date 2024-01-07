import json
import os
import random
from typing import List

from loguru import logger
from itertools import product

OUTPUTS_DIR = './outputs'
DATASET_PATH = './external/data/dataset.json'

PARAM_CNT = {
    'Baichuan2_7B_Chat': 7,
    'LLaMA2_13B_Chat': 13,
    'LLaMA2_70B_Chat': 70,
    'PHI2': 2.7,
    'gpt-3.5-turbo': 175,
}


def PARAM_CNT_TYPE(llm_param_cnt: int) -> int:
    if llm_param_cnt < 10:
        return 0
    elif llm_param_cnt < 20:
        return 1
    elif llm_param_cnt < 100:
        return 2
    else:
        return 3


TASK_TYPE = {
    "sentiment analysis": 0,
    "topic classification": 1,
    "natural language inference": 2,
    "hate speech detection": 3,
}


def TASK_DESC_LEN_TYPE(task_desc_len: int) -> int:
    if task_desc_len < 60:
        return 0
    elif task_desc_len < 80:
        return 1
    elif task_desc_len < 100:
        return 2
    elif task_desc_len < 120:
        return 3
    else:
        return 4


def QUESTION_LEN_TYPE(question_len: int) -> int:
    if question_len < 100:
        return 0
    elif question_len < 200:
        return 1
    elif question_len < 300:
        return 2
    elif question_len < 400:
        return 3
    elif question_len < 500:
        return 4
    else:
        return 5


def GRIMOIRE_LEN_TYPE(grimoire_len: int) -> int:
    if grimoire_len < 1000:
        return 0
    elif grimoire_len < 2000:
        return 1
    elif grimoire_len < 3000:
        return 2
    elif grimoire_len < 4000:
        return 3
    elif grimoire_len < 5000:
        return 4
    else:
        return 5


GRIMOIRE_TYPE = {
    "basic": 0,
    "deluxe": 1
}


def SAMPLER_TYPE(grimoire_name: str) -> int:
    if 'k-means' in grimoire_name:
        return 0
    elif 'hierarchical' in grimoire_name:
        return 1
    elif 'hard' in grimoire_name and '0.5' in grimoire_name:
        return 2
    elif 'hard' in grimoire_name and '1.0' in grimoire_name:
        return 3
    elif 'random' in grimoire_name and '0-shot' not in grimoire_name:
        return 4
    elif 'random-0-shot' in grimoire_name:
        return 5
    else:
        raise ValueError('Unknown grimoire name')


def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def raw_to_dataset(raw: dict) -> List[dict]:
    dataset = []
    for data_point in raw['results']:
        if data_point['valid'] is False:
            continue
        dataset.append({
            'llm_param_cnt_type': raw['info']['llm']['model_name'],

            'task_type': raw['info']['dataset']['task_type'],
            'task_desc_len_type': len(raw['info']['dataset']['task_description']),
            'task_desc': raw['info']['dataset']['task_description'],

            'question_len_type': len(data_point['original_data']['text']),
            'question': data_point['original_data']['text'],

            'grimoire_len_type': len(raw['info']['grimoires'][data_point['grimoire_name']]),
            'grimoire_type': data_point['grimoire_name'].split('-')[0],
            'grimoire_sampler_type': data_point['grimoire_name'],
            'grimoire': raw['info']['grimoires'][data_point['grimoire_name']],

            'label': int(data_point['correct']),
        })
    return dataset


def binning(dataset: List[dict]) -> List[dict]:
    """Binning the dataset."""
    final_dataset = []
    for data_point in dataset:
        final_dataset.append({
            'llm_param_cnt_type': PARAM_CNT_TYPE(PARAM_CNT[data_point['llm_param_cnt_type']]),

            'task_type': TASK_TYPE[data_point['task_type']],
            'task_desc_len_type': TASK_DESC_LEN_TYPE(data_point['task_desc_len_type']),
            'task_desc': data_point['task_desc'],

            'question_len_type': QUESTION_LEN_TYPE(data_point['question_len_type']),
            'question': data_point['question'],

            'grimoire_len_type': GRIMOIRE_LEN_TYPE(data_point['grimoire_len_type']),
            'grimoire_type': GRIMOIRE_TYPE[data_point['grimoire_type']],
            'grimoire_sampler_type': SAMPLER_TYPE(data_point['grimoire_sampler_type']),
            'grimoire': data_point['grimoire'],

            'label': data_point['label'],
        })
    return final_dataset


def get_dataset(outputs_dir: str) -> List[dict]:
    dataset = []
    for filename in os.listdir(outputs_dir):
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(outputs_dir, filename)
        raw = read_json_file(file_path)
        tmp_dataset = raw_to_dataset(raw)
        binned = binning(tmp_dataset)
        dataset.extend(binned)
    return dataset


def post_process(dataset: List[dict], seed: int = 22) -> List[dict]:
    random.seed(seed)
    dataset = random.sample(dataset, len(dataset))
    splitted_dataset = []
    for task_type, label in product(range(6), range(2)):
        partial = [data_point for data_point in dataset if data_point['task_type']
                   == task_type and data_point['label'] == label]
        splitted_dataset.append(partial)
    min_len = min([len(partial) for partial in splitted_dataset])
    dataset = []
    for partial in splitted_dataset:
        dataset.extend(partial[:min_len])
    return dataset


def save_json_file(file_path: str, obj) -> None:
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4)


if __name__ == '__main__':
    dataset = get_dataset(OUTPUTS_DIR)
    dataset = post_process(dataset, seed=22)
    save_json_file(DATASET_PATH, dataset)
    logger.info(f'Dataset saved at {DATASET_PATH}')
