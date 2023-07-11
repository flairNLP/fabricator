import logging
from typing import Union, List

from datasets import Dataset, QuestionAnsweringExtractive, TextClassification, TaskTemplate

from .base import LLMPrompt, ClassificationOptions

logger = logging.getLogger(__name__)


class TextLabelPrompt(LLMPrompt):
    """Prompt when label can be any form of text."""

    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        task_description: str = "Given the following examples, annotate the unlabeled example using textual format.",
        **kwargs,
    ):
        """Prompt for annotating text. Useful for tasks like question answering, summarization, translation,
        instructions, etc.

        Args:
            input_variables: List or string of input variables / column names for the fewshot / support set examples
            target_variable: Target variable / column name. The column is annotated by the LLM.
            task_description: Description of the task for the prompt prefix.
        """
        super().__init__(
            input_variables=input_variables,
            target_variable=target_variable,
            task_description=task_description,
            **kwargs,
        )


class ClassLabelPrompt(LLMPrompt):
    """Prompt when output should be a single class label."""

    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        label_options: ClassificationOptions,
        task_description: str = "Given the following classification examples, annotate the unlabeled example with a "
        "prediction that must correspond to exactly one of the following labels: "
        "{label_options}.",
        **kwargs,
    ):
        """Prompt for annotating tasks that have a single label. Useful for tasks like text classification, sentiment
        analysis, sentence similarity, etc.

        Args:
            input_variables: List or string of input variables / column names for the fewshot / support set examples
            target_variable: Target variable / column name. The column is annotated by the LLM.
            label_options: List of labels to choose from.
            task_description: Description of the task for the prompt prefix.
        """
        super().__init__(
            input_variables=input_variables,
            target_variable=target_variable,
            task_description=task_description,
            label_options=label_options,
            **kwargs,
        )


class TokenLabelPrompt(LLMPrompt):
    """Token label prompt when each token is assigned to a class."""

    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        label_options: ClassificationOptions,
        task_description: str = "Given the following token classification examples, annotate the unlabeled example in "
        "the same style as the previous ones. Choose your annotations from: {label_options}.",
        **kwargs,
    ):
        """Prompt for annotating tasks that have a token labels. Useful for tasks like named entity recognition or
        part-of-speech tagging. Similar to ClassLabelPrompt but task_description default is different.

        Args:
            input_variables: List or string of input variables / column names for the fewshot / support set examples
            target_variable: Target variable / column name. The column is annotated by the LLM.
            label_options: List of labels to choose from.
            task_description: Description of the task for the prompt prefix.
        """
        super().__init__(
            input_variables=input_variables,
            target_variable=target_variable,
            task_description=task_description,
            label_options=label_options,
            **kwargs,
        )


def infer_prompt_from_task_template(task_template: TaskTemplate):
    """Infer TextLabelPrompt or ClassLabelPrompt with correct parameters from a task template's metadata."""
    if isinstance(task_template, QuestionAnsweringExtractive):
        return TextLabelPrompt(
            input_variables=[task_template.context_column, task_template.question_column],
            target_variable="answer",  # assuming the dataset was preprocessed with preprocess_squad_format otherwise
            # dataset.task_templates[0]["answers_column"]
            task_description="Given a context and a question, generate an answer that occurs exactly and only once in "
                             "the text.",
        )
    elif isinstance(task_template, TextClassification):
        return ClassLabelPrompt(
            input_variables=[task_template.text_column],
            target_variable=task_template.label_column,
            label_options=dict(enumerate(task_template.label_schema["labels"].names))
        )
    else:
        raise ValueError(f"Automatic prompt is only supported for QuestionAnsweringExtractive and "
                         f"TextClassification tasks but not for {type(task_template)}. You need to "
                         f"specify the prompt manually.")


def infer_prompt_from_dataset(dataset: Dataset):
    """Infer TextLabelPrompt or ClassLabelPrompt with correct parameters from a dataset's metadata."""
    if not dataset.task_templates:
        raise ValueError("Dataset must have exactly one task template but there is none. You need to specify the "
                         "prompt manually.")
    elif len(dataset.task_templates) > 1:
        raise ValueError(f"Automatic prompt is only supported for datasets with exactly one task template but yours "
                         "has {len(dataset.task_templates)}. You need to specify the prompt manually.")
    else:
        return infer_prompt_from_task_template(dataset.task_templates[0])
