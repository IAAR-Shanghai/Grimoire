import random
from math import ceil
from typing import Iterable, List


class DataLoader:
    """DataLoader class for iterating over batches of data using a specified sampler.

    Methods:
        __len__(): Returns the number of batches in the DataLoader.
        __iter__(): Resets the current batch index for iteration.
        __next__(): Returns the next batch of data.

    Examples:
        >>> dataset = [{'text': 'sample1', 'ans_text': 'A'},
        ...            {'text': 'sample2', 'ans_text': 'B'},
        ...            {'text': 'sample3', 'ans_text': 'C'},
        ...            {'text': 'sample4', 'ans_text': 'A'},
        ...            {'text': 'sample5', 'ans_text': 'B'},
        ...            {'text': 'sample6', 'ans_text': 'C'},
        ...            {'text': 'sample7', 'ans_text': 'D'},
        ...            {'text': 'sample8', 'ans_text': 'A'}]

        >>> # Create a DataLoader with a specified batch size and a custom sampler
        >>> sampler = RandomSampler(dataset, seed=42, cluster_and_merge=True)
        >>> dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)

        >>> # Iterate over batches
        >>> for batch in dataloader:
        ...     print(batch)
        [{'text': 'sample4', 'ans_text': 'A'}, {'text': 'sample5', 'ans_text': 'B'}, {'text': 'sample7', 'ans_text': 'D'}, {'text': 'sample3', 'ans_text': 'C'}]
        [{'text': 'sample8', 'ans_text': 'A'}, {'text': 'sample2', 'ans_text': 'B'}, {'text': 'sample6', 'ans_text': 'C'}, {'text': 'sample1', 'ans_text': 'A'}]
    """

    def __init__(self, data: List[dict], batch_size: int, shuffle: bool = True, seed: int = None):
        """
        Args:
            data (List[dict]): The input data to load batches from.
            batch_size (int): The size of each batch.
            sampler (BaseSampler, optional): An optional sampler to specify the order of data sampling. Default is None.
        """
        random.seed(seed) if seed is not None else ...
        random.shuffle(data) if shuffle else ...
        self.data = data
        self.batch_size = batch_size

        self.current_batch = 0

    def __len__(self) -> int:
        return ceil(len(self.data) / self.batch_size)

    def __iter__(self) -> Iterable:
        self.current_batch = 0
        return self

    def __next__(self) -> List[dict]:
        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) *
                      self.batch_size, len(self.data))
        if start_idx >= end_idx:
            raise StopIteration

        batch_data = self.data[start_idx:end_idx]
        self.current_batch += 1
        return batch_data
