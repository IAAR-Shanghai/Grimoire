import importlib
from itertools import product
from typing import List, Literal, Tuple

from loguru import logger

from core.data.dataloader import DataLoader
from core.data.dataset import Dataset
from core.data.sampler.hard import KeepHardSampler
from core.data.sampler.hierarchical import HierarchicalSampler
from core.data.sampler.kmeans import KMeansSampler
from core.data.sampler.rand import RandomSampler
from core.data.sampler.similarity import SimilaritySampler
from core.evaluator.few_shot import FewShotEvaluator
from core.evaluator.grimoire_classify import GrimoireClassifyEvaluator
from core.evaluator.grimoire_rank import GrimoireRankEvaluator
from core.evaluator.with_grimoire import WithGrimoireEvaluator
from core.llm.base import BaseLLM
from core.load_conf import load_yaml_conf


DATA_CONF_PATH = './configs/data.yaml'
EXPERIMENT_CONF_PATH = './configs/experiment.yaml'
CACHE_GRIMOIRE_DIR = './.cache/grimoire_prompts'
CLASSIFIER_PATH = './.cache/classifier_model.pth'

# SentenceTransformer format!
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
EMBEDDINGS_FILENAME = 'ebd_train_all-mpnet-base-v2.pickle'
SIMILARITY_FILENAME = 'sims_all-mpnet-base-v2.pickle'

SEED = 22
EXPERIMENT_BATCH_SIZE = 500
EXPERIMENT_TIMES = 3
PROCESS = 10


def catch_all_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(repr(e))
    return wrapper


@catch_all_exceptions
def expt_few_shot(
    model: BaseLLM,
    data_name: str,
    data_conf: dict,
    few_shot_cnt: int,
    sample_method: Literal['random', 'hierarchical', 'k-means', 'similarity', 'keep-hard'],
    per_class: bool = True,
    hard_ratio: float = 0.5,
    seed: int = 22,
):
    setting = f'{sample_method}-{few_shot_cnt}-shot' \
              f'{"-hard-"+str(hard_ratio) if sample_method == "keep-hard" else ""}' \
              f'{"-per-class" if per_class else ""}'
    logger.info(setting)
    if per_class:
        if len(data_conf['data_label_list']) == 0:
            total_shot_cnt = few_shot_cnt * 4
        else:
            total_shot_cnt = few_shot_cnt * len(data_conf['data_label_list'])
    else:
        total_shot_cnt = few_shot_cnt

    train_dataset = Dataset(
        f'data/{data_name}/train.json', data_name=f'{data_name}_train').load()
    test_dataset = Dataset(f'data/{data_name}/test.json', data_name=f'{data_name}_test').resize(
        EXPERIMENT_BATCH_SIZE * EXPERIMENT_TIMES).load()

    if sample_method == 'random':
        train_sampler = RandomSampler(
            train_dataset, cnt=total_shot_cnt, cluster=per_class, identical=True, seed=seed)
    elif sample_method == 'hierarchical':
        ebds_path = f'data/{data_name}/{EMBEDDINGS_FILENAME}'
        train_sampler = HierarchicalSampler(
            train_dataset, data_conf, ebds_path, num_clusters=few_shot_cnt)
    elif sample_method == 'k-means':
        ebds_path = f'data/{data_name}/{EMBEDDINGS_FILENAME}'
        train_sampler = KMeansSampler(
            train_dataset, data_conf, ebds_path, num_clusters=few_shot_cnt, seed=seed)
    elif sample_method == 'similarity':
        sim_ranks_path = f'data/{data_name}/{SIMILARITY_FILENAME}'
        train_sampler = SimilaritySampler(
            train_dataset, sim_ranks_path, cnt=total_shot_cnt, cluster=per_class, reverse=False)
    elif sample_method == 'keep-hard':
        train_sampler = KeepHardSampler(train_dataset, data_conf, model, cnt=total_shot_cnt,
                                        hard_ratio=hard_ratio, cluster=per_class, process_num=PROCESS)

    test_dataloader = DataLoader(
        test_dataset, batch_size=EXPERIMENT_BATCH_SIZE, shuffle=True, seed=seed)

    for batch in test_dataloader:
        evaluator = FewShotEvaluator(
            model, data_conf, batch, setting, process_num=PROCESS)
        evaluator.post_init(few_shot_sampler=train_sampler)
        evaluator.run()


@catch_all_exceptions
def expt_with_grimoire(
    model: BaseLLM,
    data_name: str,
    data_conf: dict,
    few_shot_cnt: int,
    sample_method: Literal['random', 'hierarchical', 'k-means', 'similarity', 'keep-hard'],
    per_class: bool = True,
    verbose: bool = True,
    grimoire_generator: BaseLLM = None,
    grimoire_dir: str = CACHE_GRIMOIRE_DIR,
    hard_ratio: float = 0.5,
    seed: int = 22,
):
    setting = f'{sample_method}-{few_shot_cnt}-shot' \
              f'{"-hard-"+str(hard_ratio) if sample_method == "keep-hard" else ""}' \
              f'{"-per-class" if per_class else ""}' \
              f'-with-grimoire'\
              f'{"-verbose" if verbose else ""}'
    logger.info(setting)
    if per_class:
        if len(data_conf['data_label_list']) == 0:
            total_shot_cnt = few_shot_cnt * 4  # Use 4 as the default number of labels
        else:
            total_shot_cnt = few_shot_cnt * len(data_conf['data_label_list'])
    else:
        total_shot_cnt = few_shot_cnt

    train_dataset = Dataset(
        f'data/{data_name}/train.json', data_name=f'{data_name}_train').load()
    test_dataset = Dataset(f'data/{data_name}/test.json', data_name=f'{data_name}_test').resize(
        EXPERIMENT_BATCH_SIZE * EXPERIMENT_TIMES).load()

    if sample_method == 'random':
        train_sampler = RandomSampler(
            train_dataset, cnt=total_shot_cnt, cluster=per_class, identical=True, seed=seed)
    elif sample_method == 'hierarchical':
        ebds_path = f'data/{data_name}/{EMBEDDINGS_FILENAME}'
        train_sampler = HierarchicalSampler(
            train_dataset, data_conf, ebds_path, num_clusters=few_shot_cnt)
    elif sample_method == 'k-means':
        ebds_path = f'data/{data_name}/{EMBEDDINGS_FILENAME}'
        train_sampler = KMeansSampler(
            train_dataset, data_conf, ebds_path, num_clusters=few_shot_cnt, seed=seed)
    elif sample_method == 'similarity':
        sim_ranks_path = f'data/{data_name}/{SIMILARITY_FILENAME}'
        train_sampler = SimilaritySampler(
            train_dataset, sim_ranks_path, cnt=total_shot_cnt, cluster=per_class, reverse=False)
    elif sample_method == 'keep-hard':
        train_sampler = KeepHardSampler(train_dataset, data_conf, model, cnt=total_shot_cnt,
                                        hard_ratio=hard_ratio, cluster=per_class, process_num=PROCESS)

    test_dataloader = DataLoader(
        test_dataset, batch_size=EXPERIMENT_BATCH_SIZE, shuffle=True, seed=seed)

    for batch in test_dataloader:
        evaluator = WithGrimoireEvaluator(
            model, data_conf, batch, setting, process_num=PROCESS)
        evaluator.post_init(few_shot_sampler=train_sampler, verbose=verbose,
                            grimoire_generator=grimoire_generator, grimoire_dir=grimoire_dir)
        evaluator.run()


@catch_all_exceptions
def expt_grimoire_rank(
    model: BaseLLM,
    data_name: str,
    data_conf: dict,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    grimoire_generator: BaseLLM = None,
    grimoire_dir: str = CACHE_GRIMOIRE_DIR,
    filter_out_contains: List[str] = [],
    seed: int = 22,
):
    setting = 'grimoire-rank'
    logger.info(setting)

    test_dataset = Dataset(f'data/{data_name}/test.json', data_name=f'{data_name}_test').resize(
        EXPERIMENT_BATCH_SIZE * EXPERIMENT_TIMES).load()
    test_dataloader = DataLoader(
        test_dataset, batch_size=EXPERIMENT_BATCH_SIZE, shuffle=True, seed=seed)

    for batch in test_dataloader:
        evaluator = GrimoireRankEvaluator(
            model, data_conf, batch, setting, process_num=PROCESS)
        evaluator.post_init(embedding_model_name, grimoire_generator,
                            grimoire_dir, filter_out_contains=filter_out_contains)
        evaluator.run()


@catch_all_exceptions
def expt_grimoire_classify(
    model: BaseLLM,
    data_name: str,
    data_conf: dict,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    grimoire_generator: BaseLLM = None,
    grimoire_dir: str = CACHE_GRIMOIRE_DIR,
    classifier_pth_path: str = CLASSIFIER_PATH,
    filter_out_contains: List[str] = [],
    seed: int = 22,
):
    setting = 'grimoire-classify'
    logger.info(setting)

    test_dataset = Dataset(f'data/{data_name}/test.json', data_name=f'{data_name}_test').resize(
        EXPERIMENT_BATCH_SIZE * EXPERIMENT_TIMES).load()
    test_dataloader = DataLoader(
        test_dataset, batch_size=EXPERIMENT_BATCH_SIZE, shuffle=True, seed=seed)

    for batch in test_dataloader:
        evaluator = GrimoireClassifyEvaluator(
            model, data_conf, batch, setting, process_num=PROCESS)
        evaluator.post_init(embedding_model_name, grimoire_generator,
                            grimoire_dir, classifier_pth_path, filter_out_contains)
        evaluator.run()


def parse_experiment_conf(experiment_conf: dict) -> Tuple[List[BaseLLM], List[str], BaseLLM]:
    """Instantiate LLMs and provide data names based on the provided configuration.

    Args:
        experiment_conf (dict): The configuration dictionary containing 
            information about LLMs and datasets.

    Returns:
        Tuple[List[BaseLLM], List[str], BaseLLM]: A tuple containing a list 
            of instantiated LLMs, the data names and the LLM used to generate grimoires.
    """

    # ─── Get Instantiated Llms ────────────────────────────────────────────

    llm_configs = experiment_conf.get('llm', {})
    instantiated_llms = []

    for llm_type, llm_list in llm_configs.items():
        if llm_list is None:
            continue

        for llm_dict in llm_list:
            if llm_dict is None:
                continue

            generator_name = list(llm_dict.keys())[0]
            parameters = list(llm_dict.values())[0]
            llm_module_name = f"core.llm.{llm_type}"
            llm_module = importlib.import_module(llm_module_name)
            generator_class = getattr(llm_module, generator_name)

            if parameters is not None:
                instantiated_llms.append(generator_class(**parameters))
            else:
                instantiated_llms.append(generator_class())

    # ─── Get Data Names ───────────────────────────────────────────────────

    datanames = experiment_conf.get('data')

    # ─── Get The Llm Used To Generate Grimoires ───────────────────────────

    generator_conf = experiment_conf['grimoire_generator']
    generator_type = generator_conf['llm_type']
    llm_module = importlib.import_module(f"core.llm.{generator_type}")
    generator_params = generator_conf['llm_params'] or {}
    generator_class = getattr(llm_module, generator_conf['llm'])
    instantiated_generator = generator_class(**generator_params)

    return instantiated_llms, datanames, instantiated_generator


if __name__ == '__main__':
    experiment_conf = load_yaml_conf(EXPERIMENT_CONF_PATH)
    models, datanames, grimoire_generator = parse_experiment_conf(
        experiment_conf)
    data_confs = load_yaml_conf(DATA_CONF_PATH)

    for model, dataname in product(models, datanames):

        # ─── Add Experiments Here: ────────────────────────────────────

        # Baselines
        expt_few_shot(model, dataname, data_confs[dataname], few_shot_cnt=0,
                      sample_method='random', per_class=True, seed=SEED)
        expt_few_shot(model, dataname, data_confs[dataname], few_shot_cnt=4,
                      sample_method='random', per_class=False, seed=SEED)

        # Profound grimoires
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4,
                           sample_method='k-means', per_class=True, grimoire_generator=grimoire_generator, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4,
                           sample_method='hierarchical', per_class=True, grimoire_generator=grimoire_generator, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4,
                           sample_method='random', per_class=True, grimoire_generator=grimoire_generator, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4, sample_method='keep-hard',
                           per_class=True, grimoire_generator=grimoire_generator, hard_ratio=0.5, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4, sample_method='keep-hard',
                           per_class=True, grimoire_generator=grimoire_generator, hard_ratio=1.0, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=0,
                           sample_method='random', grimoire_generator=grimoire_generator, seed=SEED)

        # Simple grimoires
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4, sample_method='k-means',
                           per_class=True, verbose=False, grimoire_generator=grimoire_generator, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4, sample_method='hierarchical',
                           per_class=True, verbose=False, grimoire_generator=grimoire_generator, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4, sample_method='random',
                           per_class=True, verbose=False, grimoire_generator=grimoire_generator, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4, sample_method='keep-hard',
                           per_class=True, verbose=False, grimoire_generator=grimoire_generator, hard_ratio=0.5, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=4, sample_method='keep-hard',
                           per_class=True, verbose=False, grimoire_generator=grimoire_generator, hard_ratio=1.0, seed=SEED)
        expt_with_grimoire(model, dataname, data_confs[dataname], few_shot_cnt=0,
                           sample_method='random', verbose=False, grimoire_generator=grimoire_generator, seed=SEED)

        # Grimoire Selection
        expt_grimoire_rank(
            model, dataname, data_confs[dataname], grimoire_generator=grimoire_generator, filter_out_contains=[], seed=SEED)
        expt_grimoire_classify(
            model, dataname, data_confs[dataname], grimoire_generator=grimoire_generator, filter_out_contains=[], seed=SEED)
