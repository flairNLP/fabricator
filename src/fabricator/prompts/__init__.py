__all__ = [
    "BasePrompt",
    "infer_prompt_from_dataset",
    "infer_prompt_from_task_template"
]

from .base import BasePrompt
from .utils import infer_prompt_from_dataset, infer_prompt_from_task_template
