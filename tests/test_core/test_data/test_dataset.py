import json
import os
import unittest

from core.data.dataset import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(
            os.path.dirname(__file__), 'test_data.json')
        self.dataset = Dataset(self.test_data_path, 'test_data')

    def test_init_with_valid_file(self):
        self.assertTrue(hasattr(self.dataset, 'data'))
        self.assertTrue(hasattr(self.dataset, 'data_name'))

    def test_data_loading(self):
        with open(self.test_data_path) as f:
            expected_data = json.load(f)
        self.assertEqual(self.dataset.data, expected_data)

    def test_len(self):
        expected_length = len(self.dataset.data)
        self.assertEqual(len(self.dataset), expected_length)

    def test_getitem_with_int(self):
        first_item = self.dataset[0]
        self.assertIn(first_item, self.dataset.data)

    def test_getitem_with_slice(self):
        sliced_data = self.dataset[0:2]
        expected_sliced_data = self.dataset.data[0:2]
        self.assertEqual(sliced_data, expected_sliced_data)
