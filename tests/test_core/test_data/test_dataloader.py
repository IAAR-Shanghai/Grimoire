import unittest

from core.data.dataloader import DataLoader


class TestDataloader(unittest.TestCase):
    def setUp(self):
        self.data = [0, 1, 2, 3, 4, 5]
        self.batch_size = 2
        self.dataloader = DataLoader(self.data, self.batch_size, shuffle=False)

    def test_init(self):
        self.assertEqual(self.dataloader.data, self.data)
        self.assertEqual(self.dataloader.batch_size, self.batch_size)

    def test_len(self):
        expected_len = 3
        self.assertEqual(len(self.dataloader), expected_len)

    def test_iter(self):
        expected_batches = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]
        batches = list(self.dataloader)
        self.assertEqual(batches, expected_batches)

    def test_iter_with_single_batch(self):
        single_batch_size = 4
        self.dataloader = DataLoader(self.data, single_batch_size, shuffle=False)
        expected_batches = [
            [0, 1, 2, 3],
            [4, 5]
        ]
        batches = list(self.dataloader)
        self.assertEqual(batches, expected_batches)
