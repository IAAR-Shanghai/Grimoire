import yaml
from loguru import logger


def load_yaml_conf(yaml_path: str):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        logger.error(f"File '{yaml_path}' not found.")
    except yaml.YAMLError as e:
        logger.error(f"Unable to load YAML file '{yaml_path}': {e}")
