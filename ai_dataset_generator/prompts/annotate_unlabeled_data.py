import logging
from typing import Union, List

from .base import LLMPrompt, ClassificationOptions

logger = logging.getLogger(__name__)


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
        label_options: ClassificationOptions,
        task_description: str = "Given the following classification examples, annotate the unlabeled example with a prediction that must correspond to exactly one of the following labels: {label_options}.",
        **kwargs,
    ):
        super().__init__(
            input_variables=input_variables,
            target_variable=target_variable,
            task_description=task_description,
            label_options=label_options,
            **kwargs,
        )


class MultiLabelPrompt(LLMPrompt):
    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        label_options: ClassificationOptions,
        task_description: str = "Given the following classification examples, annotate the unlabeled example with a prediction that must correspond to zero or more of the following labels: {label_options}.",
        **kwargs,
    ):
        super().__init__(
            input_variables=input_variables,
            target_variable=target_variable,
            task_description=task_description,
            label_options=label_options,
            **kwargs,
        )


class TokenLabelPrompt(LLMPrompt):
    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        label_options: ClassificationOptions,
        task_description: str = "Given the following token classification examples, annotate the unlabeled example in the same style as the previous ones. Choose your annotations from: {label_options}.",
        **kwargs,
    ):
        super().__init__(
            input_variables=input_variables,
            target_variable=target_variable,
            task_description=task_description,
            label_options=label_options,
            **kwargs,
        )
