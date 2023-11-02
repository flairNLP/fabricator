import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from datasets import DatasetDict
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".cache")


def get_embedding_model_and_tokenizer(
    model_name_or_path: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Get embedding model and tokenizer.
    """
    model = AutoModel.from_pretrained(model_name_or_path)
    if torch.cuda.is_available():
        model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def get_classification_model_and_tokenizer(
    model_name_or_path: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Get classification model and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    if torch.cuda.is_available():
        model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

def preprocess_function(
    examples,
    tokenizer: PreTrainedTokenizer,
    task_keys: dict,
    label_to_id: dict,
):
    sentence1_key, sentence2_key = task_keys["text_column"]

    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )

    result = tokenizer(*texts, padding=True, max_length=256, truncation=True)

    if "label" in examples:
        if label_to_id is not None:
            # Map labels to IDs (not necessary for GLUE tasks)
            result["labels"] = [label_to_id[l] for l in examples["label"]]
        else:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
    return result


def get_trainloader(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    task_keys: dict,
    batch_size: int = 16,
) -> DataLoader:
    """
    Get dataloader for classification dataset.
    """
    label_column = task_keys["label_column"]
    try:
        label_to_id = dict(enumerate(dataset["train"].features[label_column].feature.names))
    except (AttributeError, KeyError):
        label_to_id = None

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task_keys": task_keys,
            "label_to_id": label_to_id
        },
        desc="Running tokenizer on dataset",
    )

    train_loader = DataLoader(
        processed_datasets["train"],
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader
