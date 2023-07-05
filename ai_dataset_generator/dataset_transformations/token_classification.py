import re
from typing import Dict, List, Tuple
from datasets import Dataset

import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.training import iob_to_biluo, biluo_tags_to_offsets, offsets_to_biluo_tags, biluo_to_iob


label_seperator = "\n"
label2entity_seperator = "->"
entity_seperator = ", "


def token_labels_to_spans(
    dataset: Dataset, token_column: str, label_column: str, id2label: Dict
) -> Tuple[Dataset, List[str]]:
    new_label_column = f"{label_column}_natural_language"
    label_options = list(set([label.replace("B-", "").replace("I-", "") for label in id2label.values()]))
    label_options.remove("O")

    def labels_to_spans(examples):
        bio_tags = [id2label[label] for label in examples[label_column]]
        bilou_tags = iob_to_biluo(bio_tags)
        doc = Doc(Vocab(), words=examples[token_column])
        offsets = biluo_tags_to_offsets(doc, bilou_tags)

        span_labels = {}
        for start, end, label in offsets:
            span_labels.setdefault(label, []).append(doc.text[start:end])

        examples[token_column] = doc.text
        span_labels = {k: entity_seperator.join(v) for k, v in span_labels.items()}
        examples[new_label_column] = label_seperator.join(
            [f"{k} {label2entity_seperator} {v}" for k, v in span_labels.items()]
        )
        return examples

    dataset = dataset.map(labels_to_spans).remove_columns(label_column).rename_column(new_label_column, label_column)

    return dataset, label_options


def spans_to_token_labels(dataset, token_column, label_column, id2label: Dict) -> Dataset:
    new_label_column = f"{label_column}_tags"
    label2id = {v: k for k, v in id2label.items()}

    def labels_to_spans(examples):
        text = examples[token_column].lower()
        str_labels = examples[label_column]

        try:
            label2entites = {}
            for annotations in str_labels.split(label_seperator):
                label, entities = annotations.split(label2entity_seperator)
                label2entites[label.strip()] = [e.strip().lower() for e in entities.split(entity_seperator)]

            spans = []
            for label, entities in label2entites.items():
                for entity in entities:
                    pattern = re.compile(re.escape(entity))
                    match = pattern.search(text)
                    if match:
                        spans.append((match.start(), match.end(), label))
                    else:
                        pass

            nlp = spacy.blank("en")
            doc = nlp(examples["tokens"])
            tags = biluo_to_iob(offsets_to_biluo_tags(doc, spans))
            tokens = [word.text for word in doc]

            examples[token_column] = tokens
            examples[new_label_column] = [label2id[tag] for tag in tags]

        except ValueError:
            examples[token_column] = []
            examples[new_label_column] = []
            return examples

        return examples

    dataset = (
        dataset.map(labels_to_spans)
        .remove_columns(label_column)
        .rename_column(new_label_column, label_column)
        .filter(lambda example: len(example[token_column]) > 0)
    )

    return dataset


def replace_token_labels(id2label: Dict, expanded_labels: Dict) -> Dict:
    replaced_id2label = {}
    for idx, tag in id2label.items():
        if tag.startswith("B-") or tag.startswith("I-"):
            prefix, label = tag.split("-", 1)
            if label in expanded_labels:
                new_label = expanded_labels[label]
                new_label_bio = f"{prefix}-{new_label}"
                replaced_id2label[idx] = new_label_bio
            else:
                replaced_id2label[idx] = tag
        else:
            replaced_id2label[idx] = tag
    return replaced_id2label
