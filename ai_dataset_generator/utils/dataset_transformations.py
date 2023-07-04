from typing import List
from datasets import Dataset, Value

def transform_labels_to_natural_language(
    dataset: Dataset,
    label_column: str,
    id2label: dict,
) -> Dataset:

    new_label_column = f"{label_column}_natural_language"

    def labels_to_natural_language(examples) -> dict:
        examples[new_label_column] = id2label[examples[label_column]]
        return examples

    dataset = dataset.map(labels_to_natural_language).remove_columns(label_column).rename_column(new_label_column, label_column)
    return dataset
