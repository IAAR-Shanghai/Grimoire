from typing import List

import torch
from sentence_transformers import SentenceTransformer
from torch import tensor
from torch.nn.functional import one_hot


class ToEmbedding:
    """Transform fields in classifier dataset to PyTorch embeddings.

    Args:
        embedding_model_name: name of the embedding model to use. SentenceTransformer format.

    Returns:
        feature: a dict of features with fields transformed to embeddings.
        label: a tensor of labels.

    """

    def __init__(self, embedding_model_name: str):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.llm_param_cnt_kinds = 4
        self.task_type_kinds = 6
        self.task_desc_len_kinds = 5
        self.question_len_kinds = 6
        self.grimoire_len_kinds = 6
        self.grimoire_type_kinds = 2
        self.grimoire_sampler_type_kinds = 6

    def __call__(self, x):
        feature, label = x[0], x[1]

        feature['llm_param_cnt_type'] = one_hot(
            tensor(feature['llm_param_cnt_type']), self.llm_param_cnt_kinds).float()

        feature['task_type'] = one_hot(
            tensor(feature['task_type']), self.task_type_kinds).float()
        feature['task_desc_len_type'] = one_hot(
            tensor(feature['task_desc_len_type']), self.task_desc_len_kinds).float()
        feature['task_desc'] = self.embedding_model.encode(
            feature['task_desc'], convert_to_tensor=True).to('cpu')

        feature['question_len_type'] = one_hot(
            tensor(feature['question_len_type']), self.question_len_kinds).float()
        feature['question'] = self.embedding_model.encode(
            feature['question'], convert_to_tensor=True).to('cpu')

        feature['grimoire_len_type'] = one_hot(
            tensor(feature['grimoire_len_type']), self.grimoire_len_kinds).float()
        feature['grimoire_type'] = one_hot(
            tensor(feature['grimoire_type']), self.grimoire_type_kinds).float()
        feature['grimoire_sampler_type'] = one_hot(tensor(
            feature['grimoire_sampler_type']), self.grimoire_sampler_type_kinds).float()
        feature['grimoire'] = self.embedding_model.encode(
            feature['grimoire'], convert_to_tensor=True).to('cpu')

        context = torch.cat([
            feature['llm_param_cnt_type'],
            feature['task_type'], feature['task_desc_len_type'], feature['task_desc'],
            feature['question_len_type'], feature['question'],
        ])
        grimoire = torch.cat([
            feature['grimoire_len_type'], feature['grimoire_type'], feature['grimoire_sampler_type'], feature['grimoire']
        ])
        label = tensor(label).float().unsqueeze(0)
        return context, grimoire, label
