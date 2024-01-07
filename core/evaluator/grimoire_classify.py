import datetime
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch import Tensor, tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.evaluator.base import BaseEvaluator
from core.llm.base import BaseLLM
from external.classifier.data.transforms import ToEmbedding
from external.classifier.model import Classifier
from external.data.dataset_preprocess import (GRIMOIRE_LEN_TYPE, GRIMOIRE_TYPE,
                                              PARAM_CNT, PARAM_CNT_TYPE,
                                              QUESTION_LEN_TYPE, SAMPLER_TYPE,
                                              TASK_DESC_LEN_TYPE, TASK_TYPE)


class GrimoireClassifyEvaluator(BaseEvaluator):
    def post_init(
        self,
        embedding_model_name: str,
        grimoire_generator: BaseLLM,
        grimoire_dir: str,
        classifier_pth_path: str,
        filter_out_contains: List[str] = [],
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.grimoires = self._read_all_grimoires(
            grimoire_dir,
            'grimoire_' + self.data_conf['data_name'] + '_' + grimoire_generator.params['model_name']
        )
        self.classifier_pth_path = classifier_pth_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Conduct filtering
        for grimoire_name in list(self.grimoires.keys()):
            for word in filter_out_contains:
                if word in grimoire_name:
                    del self.grimoires[grimoire_name]
                    break
            if '-shot-hard' in grimoire_name:
                if '-shot-hard-for-' + self.model.params['model_name'] in grimoire_name:
                    continue
                else:
                    del self.grimoires[grimoire_name]

        # Compute embeddings
        self.grimoire_ebds = self.embedding_model.encode(list(self.grimoires.values()), convert_to_tensor=True).to('cpu')
        self.task_desc_ebd = self.embedding_model.encode(self.data_conf['task_description'], convert_to_tensor=True).to('cpu')

    def evaluator_info(self) -> dict:
        return {
            'setting': self.setting_name,
            'llm': self.model.params,
            'dataset': self.data_conf,
            'grimoires': self.grimoires,
            'embedding_model': self.embedding_model_name,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def scoring(self, data_point: dict) -> dict:
        try:
            # Load data
            question_ebd = self.embedding_model.encode(question:=data_point['text'], convert_to_tensor=True).to('cpu')
            batch = []
            for (grimoire_name, grimoire), grimoire_ebd in zip(self.grimoires.items(), self.grimoire_ebds):
                input = self._prepare_single_input(
                    self.model.params['model_name'],
                    self.data_conf['task_type'], len(self.data_conf['task_description']), self.task_desc_ebd,
                    len(question), question_ebd,
                    grimoire_name, len(grimoire), grimoire_ebd
                )
                batch.append(input)
            loader = DataLoader(batch, batch_size=len(batch), shuffle=False)
            context_data, grimoire_data, label_data = next(iter(loader))
            context_data, grimoire_data = context_data.to(self.device), grimoire_data.to(self.device)

            # Load model
            model = Classifier(context_data[0].size(0), grimoire_data[0].size(0), 1)
            model.load_state_dict(torch.load(self.classifier_pth_path))
            model.to(self.device)
            model.eval()

            # Predict
            predications = model(context_data, grimoire_data)
            best_idx = torch.argmax(predications).item()
            grimoire_name = list(self.grimoires.keys())[best_idx]
            grimoire = self.grimoires[grimoire_name]

            result = self.model.classify(self.data_conf, data_point, grimoire=grimoire)
        except Exception as e:
            result = ''
            logger.warning(repr(e))
        return {
            'correct': result.lower() == data_point['ans_text'].lower(),
            'output': result,
            'grimoire_name': grimoire_name,
            'classifier_logit': predications[best_idx].item(),
            'valid': result.lower() in [label.lower() for label in self.data_conf['data_label_list']] \
                     or result.lower() in data_point['text'],
        }

    def batch_scoring(self, dataset: List[dict]) -> List[dict]:
        results = []
        for data_point in tqdm(dataset, desc=self.model.params['model_name']):
            result = self.scoring(data_point)
            results.append({**result, 'original_data': data_point})
        return results

    def compute_overall(self, valid_results: List[dict]) -> dict:
        return {
            'accuracy': sum([result['correct'] for result in valid_results]) / len(valid_results),
            'valid_num': len(valid_results),
            'grimoire_usage': dict(Counter([result['grimoire_name'] for result in valid_results])),
            'avg_classifier_logit': sum([result['classifier_logit'] for result in valid_results]) / len(valid_results),
        }

    @staticmethod
    def _prepare_single_input(
        model_name: str,
        task_type: str, task_desc_len: str, task_desc_ebd: Tensor,
        question_len: str, question_ebd: Tensor,
        grimoire_name: str, grimoire_len: int, grimoire_ebd: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        context = torch.cat([
            one_hot(tensor(PARAM_CNT_TYPE(PARAM_CNT[model_name])), 4).float(),

            one_hot(tensor(TASK_TYPE[task_type]), 6).float(),
            one_hot(tensor(TASK_DESC_LEN_TYPE(task_desc_len)), 5).float(),
            task_desc_ebd,

            one_hot(tensor(QUESTION_LEN_TYPE(question_len)), 6).float(),
            question_ebd,
        ])

        grimoire = torch.cat([
            one_hot(tensor(GRIMOIRE_LEN_TYPE(grimoire_len)), 6).float(),
            one_hot(tensor(GRIMOIRE_TYPE[grimoire_name.split('-')[0]]), 2).float(),
            one_hot(tensor(SAMPLER_TYPE(grimoire_name)), 6).float(),
            grimoire_ebd,
        ])

        label = tensor(-1).float().unsqueeze(0)
        
        return context, grimoire, label

    @staticmethod
    def _read_all_grimoires(grimoire_dir: str, grimoire_filename_start: str) -> Dict[str, str]:
        """Get all grimoires from grimoire_dir.
        
        Return:
            dict: {grimoire_name: grimoire}
        """
        grimoires = {}
        for filename in os.listdir(grimoire_dir):
            if not filename.startswith(grimoire_filename_start):
                continue
            grimoire_path = os.path.join(grimoire_dir, filename)
            with open(grimoire_path, 'r') as f:
                grimoire_dict = json.load(f)
            grimoires['deluxe-' + filename] = grimoire_dict['deluxe_grimoire']
            grimoires['basic-' + filename] = grimoire_dict['basic_grimoire']
        return grimoires
