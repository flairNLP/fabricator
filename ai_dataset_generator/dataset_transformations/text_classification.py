from typing import Dict, List, Tuple
from datasets import Dataset


def label_ids_to_textual_labels(
    dataset: Dataset,
    label_column: str,
    id2label: Dict,
) -> Tuple[Dataset, List[str]]:
    """Converts label ids to natural language labels for any classification problem with a single label such as text classification.

    Args:
        dataset (Dataset): huggingface Dataset with label ids.
        label_column (str): name of the label column.
        id2label (Dict): dictionary mapping label ids to natural language labels.

    Returns:
        Tuple[Dataset, List[str]]: huggingface Dataset with natural language labels and list of natural language labels.
    """
    new_label_column = f"{label_column}_natural_language"
    label_options = list(set([label for label in id2label.values()]))

    def labels_to_natural_language(examples):
        examples[new_label_column] = id2label[examples[label_column]]
        return examples

    dataset = (
        dataset.map(labels_to_natural_language)
        .remove_columns(label_column)
        .rename_column(new_label_column, label_column)
    )
    return dataset, label_options


def textual_labels_to_label_ids(
    dataset: Dataset,
    label_column: str,
    id2label: Dict,
) -> Dataset:
    """Converts natural language labels to label ids for any classification problem with a single label such as text classification.

    Args:
        dataset (Dataset): huggingface Dataset with natural language labels.
        label_column (str): name of the label column.
        id2label (Dict): dictionary mapping label ids to natural language labels.

    Returns:
        Dataset: huggingface Dataset with label ids.
    """
    label2id = {v: k for k, v in id2label.items()}
    new_label_column = f"{label_column}_label_id"

    def labels_to_natural_language(examples):
        examples[new_label_column] = label2id[examples[label_column]]
        return examples

    dataset = (
        dataset.map(labels_to_natural_language)
        .remove_columns(label_column)
        .rename_column(new_label_column, label_column)
    )
    return dataset


def replace_class_labels(id2label: Dict, expanded_labels: Dict) -> Dict:
    """Replaces class labels with expanded labels, i.e. label LOC should be expanded to LOCATION.
    Values of id2label need to match keys of expanded_labels.

    Args:
        id2label (Dict): mapping from label ids to label names
        expanded_labels (Dict): mapping from label names to expanded label names

    Returns:
        Dict: mapping from label ids to label names with expanded labels
    """
    replaced_id2label = {}
    for idx, tag in id2label.items():
        if tag in expanded_labels:
            replaced_id2label[idx] = expanded_labels[tag]
        else:
            replaced_id2label[idx] = tag
    return replaced_id2label
