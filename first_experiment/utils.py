from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from datasets import DatasetDict
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

PATH = Path("/glusterfs/dfs-gfs-dist/goldejon/initial-starting-point-generation")
CACHE_DIR = PATH / ".cache"


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
    id2label: dict = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Get classification model and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(id2label) if id2label is not None else None,
        id2label=id2label if id2label is not None else None,
        label2id={label: i for i, label in id2label.items()} if id2label is not None else None,
    )
    if torch.cuda.is_available():
        model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def preprocess_function(
    examples,
    tokenizer: PreTrainedTokenizer,
    task_keys: dict,
):
    sentence1_key, sentence2_key = task_keys["text_column"]

    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )

    result = tokenizer(*texts, padding=True, max_length=256, truncation=True)

    if "label" in examples:
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
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task_keys": task_keys,
        },
        desc="Running tokenizer on dataset",
    )

    train_loader = DataLoader(
        processed_datasets["train"],
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader


def get_num_epochs(batch_size: int, dataset_size: int, max_epochs: int, min_steps: int):
    total_steps = max(dataset_size// batch_size * max_epochs, min_steps)
    return total_steps * batch_size // dataset_size
