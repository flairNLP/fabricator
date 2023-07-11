from typing import Dict, List, Tuple, Union

from datasets import Dataset, DatasetDict, ClassLabel, Sequence


def get_labels_from_dataset(dataset: Union[Dataset, DatasetDict], label_column: str) -> List[str]:
    """Gets the list of labels from a huggingface Dataset.

    Args:
        dataset (Union[Dataset, DatasetDict]): huggingface Dataset
        label_column (str): name of the column with the labels

    Returns:
        List[str]: list of labels
    """
    if isinstance(dataset, DatasetDict):
        tmp_ref_dataset = dataset["train"]
    else:
        tmp_ref_dataset = dataset

    if isinstance(tmp_ref_dataset.features[label_column], ClassLabel):
        features = tmp_ref_dataset.features[label_column]
    elif isinstance(tmp_ref_dataset.features[label_column], Sequence):
        features = tmp_ref_dataset.features[label_column].feature
    else:
        raise ValueError(f"Label column {label_column} is not of type ClassLabel or Sequence.")
    return features.names


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


def convert_labels_to_texts(
    dataset: Union[Dataset, DatasetDict],
    label_column: str,
    expanded_label_mapping: Dict = None,
    return_label_options: bool = True,
) -> Tuple[Dataset | DatasetDict, list[str]] | Dataset | DatasetDict:
    """Converts labels to natural language labels for any classification problem with a single label such as text
    classification.

    Args:
        dataset (Dataset): huggingface Dataset with label ids.
        label_column (str): name of the label column.
        expanded_label_mapping (Dict, optional): dictionary mapping label ids to natural language labels.
        Defaults to None.
        return_label_options (bool, optional): whether to return the list of possible labels. Defaults to False.

    Returns:
        Tuple[Dataset, List[str]]: huggingface Dataset with natural language labels and list of natural language
        labels.
    """
    labels = get_labels_from_dataset(dataset, label_column)
    id2label = dict(enumerate(labels))

    if expanded_label_mapping is not None:
        id2label = replace_class_labels(id2label, expanded_label_mapping)

    new_label_column = f"{label_column}_natural_language"
    label_options = list(set(id2label.values()))

    def labels_to_natural_language(examples):
        examples[new_label_column] = id2label[examples[label_column]]
        return examples

    dataset = (
        dataset.map(labels_to_natural_language)
        .remove_columns(label_column)
        .rename_column(new_label_column, label_column)
    )

    if return_label_options:
        return dataset, label_options

    return dataset
