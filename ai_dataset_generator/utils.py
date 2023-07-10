import datetime
import os

from typing import Any, List


from ai_dataset_generator.prompts.base import LLMPrompt
from ai_dataset_generator.prompts import (
    GenerateUnlabeledDataPrompt,
    TextLabelPrompt,
    ClassLabelPrompt,
    TokenLabelPrompt,
)


def log_dir():
    """Returns the log directory.

    Note:
        Keep it simple for now
    """
    return os.environ.get("LOG_DIR", "./logs")


def create_timestamp_path(directory: str):
    """Returns a timestamped path for logging."""
    return os.path.join(directory, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def save_create_directory(path: str):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def infer_dummy_example(prompt_template: LLMPrompt) -> List[Any]:
    """Based on prompt and target variable, infer a dummy example."""

    prompt_type = type(prompt_template)

    if prompt_type in [GenerateUnlabeledDataPrompt, TextLabelPrompt]:
        return ["dummy text"]

    if prompt_type == ClassLabelPrompt:
        return [0]

    if prompt_type == TokenLabelPrompt:
        return [[0]]

    raise ValueError(f"Unknown prompt type {prompt_type}")
