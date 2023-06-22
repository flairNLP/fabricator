import unittest

from datasets import load_dataset

from ai_dataset_generator.dataset_loader.sampler import random_sampler, single_label_task_sampler


class TestDatasetSamplerMethods(unittest.TestCase):
    """Testcase for dataset sampler methods"""

    def setUp(self) -> None:
        self.dataset = load_dataset("imdb", split="train")

    def test_random_sampler(self):
        """Test random sampler"""
        random_sample = random_sampler(self.dataset, num_examples=10)
        self.assertEqual(len(random_sample), 10)

    def test_single_label_task_sampler(self):
        """Test single label task sampler. We use imdb which has two labels: positive and negative"""
        single_label_sample = single_label_task_sampler(self.dataset, label_column="label", num_examples=10)
        self.assertEqual(len(single_label_sample), 10)
        labels = list(single_label_sample["label"])
        self.assertEqual(len(set(labels)), 2)

    def test_single_label_task_sampler_more_examples_than_ds(self):
        """Test single label task sampler with more examples than dataset"""
        subset_dataset = self.dataset.select(range(100))
        single_label_sample = single_label_task_sampler(subset_dataset, label_column="label", num_examples=110)
        self.assertEqual(len(single_label_sample), 100)
