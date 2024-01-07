import csv
import json
import os
from collections import defaultdict
from typing import List, Tuple

from loguru import logger

from core.load_conf import load_yaml_conf


EXPERIMENT_CONF_PATH = './configs/experiment.yaml'
OUTPUTS_DIR = './outputs'
STATS_DIR = './stats'


def read_results(output: dict) -> Tuple[float, int]:
    overall = output.get('overall', {})
    return overall.get('accuracy', 0), overall.get('valid_num', 0)


def read_all_results(outputs_dir: str) -> dict:
    """Read all results from outputs."""
    results = defaultdict(list)
    for filename in os.listdir(outputs_dir):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(outputs_dir, filename), 'r') as f:
            output = json.load(f)
        acc, valid_cnt = read_results(output)
        results[filename[:-21]].append([acc, valid_cnt])
    return results


def cal_std_dev(results: list) -> float:
    """Calculate the standard deviation of the results."""
    mean = sum(results) / len(results)
    return (sum([(x - mean) ** 2 for x in results]) / len(results)) ** 0.5


def get_aggregated_results(results: dict) -> dict:
    """Calculate the average accuracy, standard deviation, and total valid counts."""
    final_results = dict()
    for setting, result in results.items():
        accs, valid_cnts = zip(*result)
        avg_acc = sum(accs) / len(accs)
        std_dev = cal_std_dev(accs)
        total_valid_cnts = sum(valid_cnts)
        final_results[setting] = [avg_acc, std_dev, total_valid_cnts]
    return final_results


def what_task_is_this_output(output_name: str, tasks: List[str]) -> str:
    """Return the task name given the output name."""
    possible_tasks = []
    for task in tasks:
        if task in output_name:
            possible_tasks.append(task)
    if len(possible_tasks) == 1:
        return possible_tasks[0]
    elif len(possible_tasks) > 1:
        return max(possible_tasks, key=len)
    else:
        return 'unknown'


def convert_to_nested_results(results: dict, data_names: List[str]) -> Tuple[dict, list]:
    """Convert to {task: {setting: [avg_acc, std_dev, total_valid_cnts]}} format."""
    nested_results = defaultdict(dict)
    all_settings = set()
    for setting, result in results.items():
        task = what_task_is_this_output(setting, data_names)
        setting = setting.replace('_'+task, '')
        all_settings.add(setting)
        nested_results[task][setting] = result
    return nested_results, all_settings


def get_all_llms(llm_conf: dict) -> List[str]:
    llms = (llm_conf.get('api', []) or []) + (llm_conf.get('remote', [])
                                              or []) + (llm_conf.get('local', []) or [])
    llm_names = []
    for llm in llms:
        llm_class_name = list(llm.keys())[0]
        if llm[llm_class_name] is not None:
            llm_names.append(llm[llm_class_name]['model_name'])
        else:
            llm_names.append(llm_class_name)
    return llm_names


def save_stats(nested_results: dict, data_names: List[str], which_model: str, all_settings: list, avg_acc_only: bool = True) -> None:
    """Save the results to a csv file."""
    # Four columns here, setting, task-avg_acc, task-std_dev, task-total_valid_cnts
    os.makedirs(STATS_DIR, exist_ok=True)
    path = os.path.join(STATS_DIR, f'stats-{which_model}.csv')
    with open(path, 'w') as f:
        writer = csv.writer(f)
        header = ['setting']
        for data_name in data_names:
            header.extend(
                [data_name+' Avg. Acc.'] if avg_acc_only else
                [data_name+' Avg. Acc.', data_name +
                    ' Std. Dev.', data_name+' Total Valid']
            )
        writer.writerow(header)
        for setting in all_settings:
            if which_model not in setting:
                continue
            row = [setting]
            for data_name in data_names:
                setting_metric = nested_results.get(data_name, dict())
                row.extend(
                    setting_metric.get(
                        setting, ['-', '-', '-'])[: (1 if avg_acc_only else 3)]
                )
            writer.writerow(row)
    logger.info(f'{path} saved.')


if __name__ == '__main__':

    # Read all results from outputs
    results = read_all_results(OUTPUTS_DIR)

    # Calculate the average accuracy, standard deviation, and total valid counts
    aggregated = get_aggregated_results(results)

    # Convert to {task: {setting: [avg_acc, std_dev, total_valid_cnts]}} format
    data_names = list(load_yaml_conf(EXPERIMENT_CONF_PATH)['data'])
    nested_results, all_settings = convert_to_nested_results(
        aggregated, data_names)

    # Save the results to csv files
    llm_conf = load_yaml_conf(EXPERIMENT_CONF_PATH)['llm']
    for llm in get_all_llms(llm_conf):
        save_stats(nested_results, data_names, llm,
                   all_settings, avg_acc_only=False)
