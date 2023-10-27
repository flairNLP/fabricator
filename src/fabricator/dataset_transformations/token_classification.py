import re
from typing import Dict, List, Tuple, Union
from datasets import Dataset, Sequence

from loguru import logger

# These are fixed for encoding the prompt and decoding the output of the LLM
SPAN_ANNOTATION_TEMPLATE = "{entity} is {label} entity."
SPAN_ANNOTATION_REGEX = r'(.+) is (.+) entity\.'


def convert_token_labels_to_spans(
    dataset: Dataset,
    token_column: str,
    label_column: str,
    expanded_label_mapping: Dict = None,
    return_label_options: bool = False
) -> Union[Dataset, Tuple[Dataset, List[str]]]:
    """Converts token level labels to spans. Useful for NER tasks to prompt the LLM with natural language labels.

    Args:
        dataset (Dataset): huggingface Dataset with token level labels
        token_column (str): name of the column with the tokens
        label_column (str): name of the column with the token level labels
        expanded_label_mapping (Dict): mapping from label ids to label names. Defaults to None.
        return_label_options (bool): whether to return a list of all possible annotations of the provided dataset

    Returns:
        Tuple[Dataset, List[str]]: huggingface Dataset with span labels and list of possible labels for the prompt
    """
    if expanded_label_mapping:
        if not len(expanded_label_mapping) == len(dataset.features[label_column].feature.names):
            raise ValueError(
                f"Length of expanded label mapping and original number of labels in dataset do not match.\n"
                f"Original labels: {dataset.features[label_column].feature.names}"
                f"Expanded labels: {list(expanded_label_mapping.values())}"
            )
        id2label = expanded_label_mapping
    elif isinstance(dataset.features[label_column], Sequence):
        id2label = dict(enumerate(dataset.features[label_column].feature.names))
    else:
        raise ValueError("Labels must be a Sequence feature or expanded_label_mapping must be provided.")

    span_column = "span_annotations"

    def labels_to_spans(example):
        span_annotations = [id2label.get(label).replace("B-", "").replace("I-", "") for label in example[label_column]]

        annotations_for_prompt = ""

        current_entity = None
        current_entity_type = None
        for idx, span_annotation in enumerate(span_annotations):
            if span_annotation == "O":
                if current_entity is not None:
                    annotations_for_prompt += SPAN_ANNOTATION_TEMPLATE.format(entity=current_entity,
                                                                              label=current_entity_type) + "\n"
                    current_entity = None
                    current_entity_type = None
                continue
            if current_entity is None:
                current_entity = example[token_column][idx]
                current_entity_type = span_annotation
                continue
            if current_entity_type == span_annotation:
                current_entity += " " + example[token_column][idx]
            else:
                annotations_for_prompt += SPAN_ANNOTATION_TEMPLATE.format(entity=current_entity,
                                                                          label=current_entity_type) + "\n"
                current_entity = example[token_column][idx]
                current_entity_type = span_annotation

        if current_entity is not None:
            annotations_for_prompt += SPAN_ANNOTATION_TEMPLATE.format(entity=current_entity,
                                                                      label=current_entity_type) + "\n"

        example[token_column] = " ".join(example[token_column])
        example[span_column] = annotations_for_prompt.rstrip("\n")
        return example

    dataset = dataset.map(labels_to_spans).remove_columns(label_column).rename_column(span_column, label_column)

    if return_label_options:
        # Spans have implicit BIO format, so sequences come in BIO format, we can ignore it
        label_options = list({label.replace("B-", "").replace("I-", "") for label in id2label.values()})

        # Ignore "outside" tokens
        if "O" in label_options:
            label_options.remove("O")

        return dataset, label_options

    return dataset


def convert_spans_to_token_labels(
        dataset: Dataset,
        token_column: str,
        label_column: str,
        id2label: Dict,
        annotate_identical_words: bool = False
) -> Dataset:
    """Converts span level labels to token level labels.
    First, the function extracts all entities with its annotated types.
    Second, if annotations are present, the function converts them to a tag sequence in BIO format.
    If not present, simply return tag sequence of O-tokens.
    This is useful for NER tasks to decode the output of the LLM.

    Args:
        dataset (Dataset): huggingface Dataset with span level labels
        token_column (str): name of the column with the tokens
        label_column (str): name of the column with the span level labels
        id2label (Dict): mapping from label ids to label names
        annotate_identical_words (bool): whether to annotate all identical words in a sentence with a found entity
        type

    Returns:
        Dataset: huggingface Dataset with token level labels in BIO format
    """
    new_label_column = "sequence_tags"
    lower_label2id = {label.lower(): idx for idx, label in id2label.items()}

    def labels_to_spans(example):
        span_annotations = example[label_column].split("\n")

        ner_tag_tuples = []

        for span_annotation in span_annotations:
            matches = re.match(SPAN_ANNOTATION_REGEX, span_annotation)
            if matches:
                matched_entity = matches.group(1)
                matched_label = matches.group(2)

                span_tokens = matched_entity.split(" ")
                span_labels = ["B-" + matched_label if idx == 0 else "B-" + matched_label.lower()
                               for idx, _ in enumerate(span_tokens)]

                for token, label in zip(span_tokens, span_labels):
                    label_id = lower_label2id.get(label.lower())
                    if label_id is None:
                        logger.info(f"Entity {token} with label {label} is not in id2label: {id2label}.")
                    else:
                        ner_tag_tuples.append((token, label_id))
            else:
                pass

        if ner_tag_tuples:
            lower_tokens = example[token_column].lower().split(" ")
            # initialize all tokens with O type
            ner_tags = [0] * len(lower_tokens)
            for reference_token, entity_type_id in ner_tag_tuples:
                if lower_tokens.count(reference_token.lower()) == 0:
                    logger.info(
                        f"Entity {reference_token} is not found or occurs more than once: {lower_tokens}. "
                        f"Thus, setting label to O."
                    )
                elif lower_tokens.count(reference_token.lower()) > 1:
                    if annotate_identical_words:
                        insert_at_idxs = [index for index, value in enumerate(lower_tokens)
                                            if value == reference_token.lower()]
                        for insert_at_idx in insert_at_idxs:
                            ner_tags[insert_at_idx] = entity_type_id
                    else:
                        logger.info(
                            f"Entity {reference_token} occurs more than once: {lower_tokens}. "
                            f"Thus, setting label to O."
                        )
                else:
                    insert_at_idx = lower_tokens.index(reference_token.lower())
                    ner_tags[insert_at_idx] = entity_type_id
        else:
            ner_tags = [0] * len(example[token_column].split(" "))

        example[token_column] = example[token_column].split(" ")
        example[new_label_column] = ner_tags

        return example

    dataset = (
        dataset.map(labels_to_spans)
        .remove_columns(label_column)
        .rename_column(new_label_column, label_column)
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
