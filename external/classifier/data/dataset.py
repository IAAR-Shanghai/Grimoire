import json
import os
import pickle
from copy import deepcopy
from typing import List

from torch.utils.data import Dataset
from tqdm import tqdm


class ClassifierDataset(Dataset):
    def __init__(self, dataset_path: str, transform: callable = None, cache_dir: str = './.cache/'):
        if transform is None:
            self.dataset = self._read_json_file(dataset_path)
            print('Dataset loaded')
        else:
            pickle_dataset_path = cache_dir + 'dataset.pickle'
            if os.path.exists(pickle_dataset_path):
                with open(pickle_dataset_path, 'rb') as f:
                    self.dataset = pickle.load(f)
                print('Dataset loaded from cache')
            else:
                os.makedirs(cache_dir, exist_ok=True)
                self.dataset = self.transform_dataset(
                    self._read_json_file(dataset_path), transform)
                with open(pickle_dataset_path, 'wb') as f:
                    pickle.dump(self.dataset, f)
                print('Dataset transformed and cached')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def _read_json_file(file_path: str) -> List[dict]:
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def transform_dataset(dataset: List[dict], transform: callable) -> List[dict]:
        if transform is None:
            return dataset

        transformed_dataset = []
        for data_point in tqdm(dataset, desc='Transforming dataset'):
            copied_data_point = deepcopy(data_point)
            del copied_data_point['label']

            feature = copied_data_point
            label = data_point['label']
            sample = (feature, label)
            transformed_dataset.append(transform(sample))

        return transformed_dataset
