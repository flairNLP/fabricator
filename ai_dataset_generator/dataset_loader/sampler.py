"""Sampling methods"""
import logging
import random

from datasets import ClassLabel, Dataset, Value

logger = logging.getLogger(__name__)


def random_sampler(dataset: Dataset, num_examples: int) -> Dataset:
    """Random sampler"""
    return dataset.select(random.sample(range(len(dataset)), num_examples))


def single_label_task_sampler(dataset: Dataset, label_column: str, num_examples: int) -> Dataset:
    """Sampler for single label tasks, like text classification

    Args:
        dataset: Dataset
        label_column: Name of the label column
        num_examples: Number of examples to sample

    Approach: 
        num_examples > len(dataset): Samples all examples
        num_examples < len(dataset): Samples at least one example per label
        num_examples < len(dataset.features): Samples only num_examples and notify
    """

    if num_examples > len(dataset):
        return dataset
    
    if "train" in dataset:
        dataset = dataset["train"]

    # Note: PHA: We dont even need to know all labels beforehand
    # features = dataset.features

    # if label_column not in features:
    #     raise ValueError(f"Label column {label_column} not found in dataset")

    # if isinstance(features[label_column], Value):
    #     logger.info(f"Label column {label_column} is of type Value. Inferring labels from dataset")
    #     class_labels = dataset.class_encode_column(label_column).features[label_column].names
    # elif isinstance(features[label_column], ClassLabel):
    #     class_labels = features[label_column].names
    # else:
    #     raise ValueError(f"Label column {label_column} is of type {type(features[label_column])}. Expected Value or ClassLabel")

    # idx_to_label = {idx: label for idx, label in enumerate(class_labels)}

    unique_classes_sampled = set()
    total_examples_sampled = 0

    sampled_indices = []

    while total_examples_sampled < num_examples:
        # Lets try to be as random and possible and sample from the entire dataset
        idx = random.sample(range(len(dataset)), 1)[0]
        sample = dataset.select([idx])[0]
        label = sample[label_column]

        # First sample at least one example per label
        if label not in unique_classes_sampled:
            unique_classes_sampled.add(label)
            sampled_indices.append(idx)
            total_examples_sampled += 1

        # Further sample if we collected at least one example per label
        elif len(sampled_indices) < num_examples:
            sampled_indices.append(idx)
            total_examples_sampled += 1

    return dataset.select(sampled_indices)

