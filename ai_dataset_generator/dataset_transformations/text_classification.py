from typing import Dict, List, Tuple
from datasets import Dataset


def label_ids_to_textual_labels(
    dataset: Dataset,
    label_column: str,
    id2label: Dict,
) -> Tuple[Dataset, List[str]]:
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
