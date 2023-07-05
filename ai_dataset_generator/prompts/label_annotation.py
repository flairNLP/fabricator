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
        """Prompt for annotating text. Useful for tasks like quesion answering, summarization, translation, instructions, etc.
        
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
    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        label_options: ClassificationOptions,
        task_description: str = "Given the following classification examples, annotate the unlabeled example with a prediction that must correspond to exactly one of the following labels: {label_options}.",
        **kwargs,
    ):
        """Prompt for annotating tasks that have a single label. Useful for tasks like text classification, sentiment analysis, etc.
        
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
    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: str,
        label_options: ClassificationOptions,
        task_description: str = "Given the following token classification examples, annotate the unlabeled example in the same style as the previous ones. Choose your annotations from: {label_options}.",
        **kwargs,
    ):
        """Prompt for annotating tasks that have a token labels. Useful for tasks like named entity recognition or part-of-speech tagging.
        Similar to ClassLabelPrompt but task_description default is different.

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
