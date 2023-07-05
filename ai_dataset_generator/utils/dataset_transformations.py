from typing import Dict, List, Tuple
from datasets import Dataset

from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.training import iob_to_biluo, biluo_tags_to_offsets


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


def transform_token_labels_to_spans(dataset, token_column, label_column, id2label: Dict) -> Tuple[Dataset, List[str]]:
    new_label_column = f"{label_column}_natural_language"
    label_options = list(set([label.replace("B-", "").replace("I-", "") for label in id2label.values()]))

    def labels_to_spans(examples):
        bio_tags = [id2label[label] for label in examples[label_column]]
        bilou_tags = iob_to_biluo(bio_tags)
        doc = Doc(Vocab(), words=examples[token_column])
        offsets = biluo_tags_to_offsets(doc, bilou_tags)

        span_labels = {}
        for start, end, label in offsets:
            span_labels.setdefault(label, []).append(doc.text[start:end])

        examples[token_column] = doc.text
        span_labels = {k: ", ".join(v) for k, v in span_labels.items()}
        examples[new_label_column] = "\n".join([f"{k} -> {v}" for k, v in span_labels.items()])
        return examples

    dataset = dataset.map(labels_to_spans).remove_columns(label_column).rename_column(new_label_column, label_column)

    return dataset, label_options
