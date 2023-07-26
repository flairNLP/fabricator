import re
from typing import Dict, List, Tuple
from collections import defaultdict
from datasets import Dataset, Sequence

from tqdm import tqdm
import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.training import iob_to_biluo, biluo_tags_to_offsets, offsets_to_biluo_tags, biluo_to_iob

# These are fixed for encoding the prompt and decoding the output of the LLM
LABEL_SEPARATOR = "\n"
LABEL2ENTITY_SEPARATOR = "->"
ENTITY_SEPARATOR = ", "


def convert_token_labels_to_spans(
    dataset: Dataset, token_column: str, label_column: str, expanded_label_mapping: Dict = None
) -> Tuple[Dataset, List[str]]:
    """Converts token level labels to spans. Useful for NER tasks to prompt the LLM with natural language labels.

    Args:
        dataset (Dataset): huggingface Dataset with token level labels
        token_column (str): name of the column with the tokens
        label_column (str): name of the column with the token level labels
        expanded_label_mapping (Dict): mapping from label ids to label names. Defaults to None.

    Returns:
        Tuple[Dataset, List[str]]: huggingface Dataset with span labels and list of possible labels for the prompt
    """
    if expanded_label_mapping:
        id2label = expanded_label_mapping
    elif isinstance(dataset.features[label_column], Sequence):
        id2label = dict(enumerate(dataset.features[label_column].feature.names))
    else:
        raise ValueError("Labels must be a Sequence feature or expanded_label_mapping must be provided.")

    new_label_column = f"{label_column}_natural_language"
    label_options = list({label.replace("B-", "").replace("I-", "") for label in id2label.values()})
    if "O" in label_options:
        label_options.remove("O")

    def labels_to_spans(examples):
        bio_tags = [id2label[label] for label in examples[label_column]]
        bilou_tags = iob_to_biluo(bio_tags)
        doc = Doc(Vocab(), words=examples[token_column])
        offsets = biluo_tags_to_offsets(doc, bilou_tags)

        span_labels = defaultdict(list)
        for start, end, label in offsets:
            span_labels[label].append(doc.text[start:end])

        examples[token_column] = doc.text
        span_labels = {k: ENTITY_SEPARATOR.join(v) for k, v in span_labels.items()}
        examples[new_label_column] = LABEL_SEPARATOR.join(
            [f"{k} {LABEL2ENTITY_SEPARATOR} {v}" for k, v in span_labels.items()]
        )
        return examples

    dataset = dataset.map(labels_to_spans).remove_columns(label_column).rename_column(new_label_column, label_column)

    return dataset, label_options


def convert_spans_to_token_labels(dataset, token_column, label_column, id2label: Dict) -> Dataset:
    """Converts span level labels to token level labels. This is useful for NER tasks to decode the output of the LLM.
    #TODO: this is very slow for large datasets. We should remove dependency from spacy at some point.

    Args:
        dataset (Dataset): huggingface Dataset with span level labels
        token_column (str): name of the column with the tokens
        label_column (str): name of the column with the span level labels
        id2label (Dict): mapping from label ids to label names

    Returns:
        Dataset: huggingface Dataset with token level labels in BIO format
    """
    new_label_column = f"{label_column}_tags"
    label2id = {v: k for k, v in id2label.items()}
    nlp = spacy.blank("en")

    def labels_to_spans(examples):
        texts = examples[token_column]
        str_labels = examples[label_column]
        # goal list of lists of tuples (start, end, label)

        tokens = []
        bio_tags = []
        for text, str_label in tqdm(zip(texts, str_labels), desc="Converting spans to token labels"):
            spans = []

            if not str_label:
                bio_tags.append([])
                tokens.append([])
                continue

            try:
                for label_and_entities in str_label.split(LABEL_SEPARATOR):
                    label, entities = label_and_entities.split(LABEL2ENTITY_SEPARATOR)
                    label = label.strip()
                    entities = [entity.strip().lower() for entity in entities.split(ENTITY_SEPARATOR)]
                    for entity in set(entities):
                        pattern = re.compile(re.escape(entity))
                        matches = pattern.finditer(text.lower())
                        for start, end in [(match.start(), match.end()) for match in matches]:
                            spans.append((start, end, label))
            except ValueError:
                bio_tags.append([])
                tokens.append([])
                continue

            doc = nlp(text)

            try:
                tags = biluo_to_iob(offsets_to_biluo_tags(doc, spans))
                words = [word.text for word in doc]
                if not len(tags) == len(words) or len(tags) == 0 or len(words) == 0:
                    tags = []
                    words = []
                bio_tags.append(tags)
                tokens.append(words)
            except ValueError:
                bio_tags.append([])
                tokens.append([])
                continue

        examples[token_column] = tokens
        examples[new_label_column] = [[label2id[tag] for tag in tags] for tags in bio_tags]

        return examples

    dataset = (
        dataset.map(labels_to_spans, batched=True)
        .remove_columns(label_column)
        .rename_column(new_label_column, label_column)
        .filter(lambda example: len(example[token_column]) > 0)
    )

    return dataset


def replace_token_labels(id2label: Dict, expanded_labels: Dict) -> Dict:
    """Replaces token level labels with expanded labels, i.e. label PER should be expanded to PERSON.
    Values of id2label need to match keys of expanded_labels.

    Args:
        id2label (Dict): mapping from label ids to label names
        expanded_labels (Dict): mapping from label names to expanded label names

    Returns:
        Dict: mapping from label ids to label names with expanded labels
    """
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
