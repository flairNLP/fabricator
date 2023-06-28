import unittest

from datasets import load_dataset

from ai_dataset_generator.samplers.sampler import random_sampler, single_label_task_sampler, ml_mc_sampler


def _flatten(l):
    return [item for sublist in l for item in sublist]


class TestDatasetSamplerMethodsSingleLabel(unittest.TestCase):
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


class TestDatasetSamplerMethodsMultiLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset("conll2003", split="train")

    def test_ml_mc_sampler(self):
        subset_dataset = ml_mc_sampler(self.dataset, labels_column="pos_tags", num_examples=10)
        label_idxs = list(range(len(self.dataset.features["pos_tags"].feature.names)))
        self.assertEqual(len(subset_dataset), 10)

        tags = set(_flatten([sample["pos_tags"] for sample in subset_dataset]))

        # We do not guarantee, that all tags are contained in sampled examples
        self.assertLessEqual(len(tags), len(label_idxs))

    # def test_ml_mc_sampler_ensure_all_labels(self):
    #     subset_dataset = ml_mc_sampler(self.dataset, labels_column="pos_tags", num_examples=-1)
    #     label_idxs = list(range(len(self.dataset.features["pos_tags"].feature.names)))
    #     self.assertEqual(len(subset_dataset), 10)
    #
    #     tags = set(_flatten([sample["pos_tags"] for sample in subset_dataset]))
    #
    #     # We do not guarantee, that all tags are contained in sampled examples
    #     self.assertEqual(len(tags), len(label_idxs))
