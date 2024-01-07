import json
import os
from typing import Any, List, Union


class Dataset:
    def __init__(self, path_or_data: Union[str, list], data_name: str):
        """
        Args:
            path_or_data (Union[str, list]): The path to the dataset file or the dataset.
            data_name (str): The name of the dataset.
        """
        if isinstance(path_or_data, str):
            self._read_from_file(path_or_data)
        elif isinstance(path_or_data, list):
            self.data = path_or_data
        else:
            raise ValueError("Invalid input for 'path_or_data'. Provide either a file path or a list of data.")

        self.data_name = data_name

    def load(self) -> List[dict]:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: Union[int, slice]) -> Union[Any, List[Any]]:
        return self.data[key]

    def _read_from_file(self, path: str):
        """Read data from a file."""
        if os.path.isfile(path):
            with open(path) as f:
                self.data = json.load(f)
        else:
            raise FileNotFoundError(f"The file '{path}' does not exist.")

    # ─── Pre-processers ───────────────────────────────────────────────────

    def resize(self, size: int) -> 'Dataset':
        """Resize the dataset to a given size.
        
        Args:
            size (int): The new size of the dataset.
        
        Returns:
            Dataset: A new Dataset object containing the resized dataset.
        """
        times = size // len(self.data)
        remainder = size % len(self.data)
        
        self.data = self.data * times + self.data[ : remainder]
        return self
