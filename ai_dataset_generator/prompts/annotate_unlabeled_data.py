import logging
from typing import Union, List, Dict

from .base import LLMPrompt

logger = logging.getLogger(__name__)

ClassificationLabels = List[str]
ID2Label = Dict[int, str]
ClassificationOptions = Union[ClassificationLabels, ID2Label]


class TextLabelPrompt(LLMPrompt):
    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        task_description: str = "Given the following examples, annotate the unlabeled example using textual format.",
        **kwargs,
    ):
        super().__init__(
            input_variables=input_variables,
            target_variable=target_variable,
            task_description=task_description,
            **kwargs,
        )


class SingleLabelPrompt(LLMPrompt):
    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        classification_label_options: ClassificationOptions,
        task_description: str = "Given the following classification examples, annotate the unlabeled example with a prediction that must correspond to exactly one of the following labels: {classification_label_options}.",
    ):
        if not "{classification_label_options}" in task_description:
            logger.warning(
                "{classification_label_options} is not found in the task_description. If you want to limit your answers to these information, make sure to include {classification_label_options}."
            )

        if isinstance(classification_label_options, dict):
            formatted_labels = ", ".join([f"{k}: {v}" for k, v in classification_label_options.items()])
        elif isinstance(classification_label_options, list):
            formatted_labels = ", ".join(classification_label_options)
        else:
            raise AttributeError("Type of label options must be either Dict[int, str] or List[str]")

        try:
            task_description = task_description.format(classification_label_options=formatted_labels)
        except KeyError:
            raise KeyError(
                "The provided task description cannot be formatted with the variable 'classification_label_options'. "
                "You need to include {classification_label_options} in the task_description string to limit your prompt output to these labels. "
                "For example: task_description='[...] limit your answer to: {classification_label_options}.'"
            )

        super().__init__(
            input_variables=input_variables, target_variable=target_variable, task_description=task_description
        )


class MultiLabelPrompt(LLMPrompt):
    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        classification_label_options: ClassificationOptions,
        task_description: str = "Given the following classification examples, annotate the unlabeled example with a prediction that must correspond to zero or more of the following labels: {classification_label_options}.",
    ):
        if not "{classification_label_options}" in task_description:
            logger.warning(
                "{classification_label_options} is not found in the task_description. If you want to limit your answers to these information, make sure to include {classification_label_options}."
            )

        if isinstance(classification_label_options, dict):
            formatted_label_options = ", ".join([f"{k}: {v}" for k, v in classification_label_options.items()])
        elif isinstance(classification_label_options, list):
            formatted_label_options = ", ".join(classification_label_options)
        else:
            raise AttributeError("Type of label options must be either Dict[int, str] or List[str]")

        try:
            task_description = task_description.format(classification_label_options=formatted_label_options)
        except KeyError:
            raise KeyError(
                "The provided task description cannot be formatted with the variable 'classification_label_options'. "
                "You need to include {classification_label_options} in the task_description string to limit your prompt output to these labels. "
                "For example: task_description='[...] limit your answer to: {classification_label_options}.'"
            )

        super().__init__(
            input_variables=input_variables, target_variable=target_variable, task_description=task_description
        )


class TokenLabelPrompt(LLMPrompt):
    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        token_label_options: ClassificationLabels,
        task_description: str = "Given the following token classification examples, list all {token_label_options} in the unlabeled example.",
    ):
        if not "{token_label_options}" in task_description:
            logger.warning(
                "{token_label_options} is not found in the task_description. If you want to limit your answers to these information, make sure to include {token_label_options}."
            )

        if isinstance(token_label_options, dict):
            formatted_label_options = ", ".join([f"{k}: {v}" for k, v in token_label_options.items()])
        elif isinstance(token_label_options, list):
            formatted_label_options = ", ".join(token_label_options)
        else:
            raise AttributeError("Type of label options must be either Dict[int, str] or List[str]")

        try:
            task_description = task_description.format(token_label_options=formatted_label_options)
        except KeyError:
            raise KeyError(
                "The provided task description cannot be formatted with the variable 'token_label_options'. "
                "You need to include {token_label_options} in the task_description string to limit your prompt output to these labels. "
                "For example: task_description='[...] limit your answer to: {token_label_options}.'"
            )

        super().__init__(
            input_variables=input_variables, target_variable=target_variable, task_description=task_description
        )
