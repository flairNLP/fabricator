"""Sampling methods

NOTE: All methods do not ensure, that all labels are contained in the samples.
TODO: Implement mechanism: like num_examples == -1 -> infer labels and sample
      as long as we do not have them all)

"""
import random
from typing import Dict, List, Set, Union

from datasets import ClassLabel, Dataset, Sequence, Value
from loguru import logger
from tqdm import tqdm


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

    pbar = tqdm(total=num_examples, desc="Sampling")

    class_labels = _infer_class_labels(dataset, label_column)
    num_classes = len(class_labels)

    unique_classes_sampled = set()
    total_examples_sampled = 0

    sampled_indices = []

    while total_examples_sampled < num_examples:
        # Lets try to be as random and possible and sample from the entire dataset
        idx = random.sample(range(len(dataset)), 1)[0]

        # Pass already sampled idx
        if idx in sampled_indices:
            continue

        sample = dataset.select([idx])[0]
        label = sample[label_column]

        # First sample at least one example per label
        if label not in unique_classes_sampled:
            unique_classes_sampled.add(label)
            sampled_indices.append(idx)
            total_examples_sampled += 1
            pbar.update(1)

        # Further sample if we collected at least one example per label
        elif len(sampled_indices) < num_examples and len(unique_classes_sampled) == num_classes:
            sampled_indices.append(idx)
            total_examples_sampled += 1
            pbar.update(1)

    return dataset.select(sampled_indices)


def ml_mc_sampler(dataset: Dataset, labels_column: str, num_examples: int) -> Dataset:
    """Multi label multi class sampler

    Args:
        dataset: Dataset
        label_column: Name of the label column
        num_examples: Number of examples to sample, if -1 sample as long as subset does not contain every label

    """

    if num_examples > len(dataset):
        return dataset

    if "train" in dataset:
        dataset = dataset["train"]

    total_labels = _infer_class_labels(dataset, labels_column)
    num_classes = len(total_labels)

    # Because of random sampling we do not ensure, that we ever sampled all examples
    # Nor do we know if all labels are present. We therefore use a max try counter
    # So we dont get stuck in infinite while loop
    if num_examples == -1:
        max_tries = 2 * len(dataset)
    else:
        max_tries = -1

    tries = 0

    pbar = tqdm(total=num_examples, desc="Sampling")

    unique_classes_sampled: Set[str] = set()
    total_examples_sampled = 0

    sampled_indices = []

    while len(unique_classes_sampled) < len(total_labels):
        # Lets try to be as random and possible and sample from the entire dataset
        idx = random.sample(range(len(dataset)), 1)[0]

        # Pass already sampled idx
        if idx in sampled_indices:
            continue

        sample = dataset.select([idx])[0]

        labels = sample[labels_column]

        if not isinstance(labels, list):
            labels = [labels]

        labels_found = [label for label in labels if label not in unique_classes_sampled]

        # Check if current sample contains labels not found yet
        complements = _relative_complements(labels_found, unique_classes_sampled)

        if len(complements) > 0:
            unique_classes_sampled.update(complements)
            sampled_indices.append(idx)
            total_examples_sampled += 1
            pbar.update(1)

        # Further sample if we collected at least one example per label
        elif len(sampled_indices) < num_examples and len(unique_classes_sampled) == num_classes:
            sampled_indices.append(idx)
            total_examples_sampled += 1
            pbar.update(1)

        if num_examples != -1 and total_examples_sampled == num_examples:
            break

        tries += 1
        if tries == max_tries:
            logger.info("Stopping sample. Max tries(={}) exceeded.", max_tries)
            break

    return dataset.select(sampled_indices)


def _infer_class_labels(dataset: Dataset, label_column: str) -> Dict[int, str]:
    """Infer the total set of labels"""
    features = dataset.features

    if label_column not in features:
        raise ValueError(f"Label column {label_column} not found in dataset")

    if isinstance(features[label_column], Value):
        logger.info("Label column {} is of type Value. Inferring labels from dataset", label_column)
        class_labels = dataset.class_encode_column(label_column).features[label_column].names
    elif isinstance(features[label_column], ClassLabel):
        class_labels = features[label_column].names
    elif isinstance(features[label_column], Sequence):
        class_labels = features[label_column].feature.names
    else:
        raise ValueError(
            f"Label column {label_column} is of type {type(features[label_column])}. Expected Value, "
            f"ClassLabel or Sequence"
        )

    return dict(enumerate(class_labels))


def _relative_complements(list1: List, list2: Union[List, Set]) -> Set:
    """a \\ b"""
    return set(list1) - set(list2)
