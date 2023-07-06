from typing import Union, List
from .base import LLMPrompt


class GenerateUnlabeledDataPrompt(LLMPrompt):
    """Prompt for generating unlabeled data."""

    def __init__(
        self,
        input_variables: Union[List[str], str],
        task_description: str = "Generate similar texts. Your answer must be in the same format as the examples.",
        **kwargs,
    ):
        """Prompt for generating unlabeled data.

        Args:
            input_variables: Input variables / column name to use for generating unlabeled data.
            task_description: Description of the task for the prompt prefix.

        Raises:
            AssertionError: If more than one input variable is passed.
            AssertionError: If a target variable is passed.
        """
        # Check if creating unlabeled data, that only one input_variable is passed
        assert len(input_variables) == 1, (
            "When generating new unlabeled data, you must use exactly one input variable to generate similar text "
            "specifically associated with that variable."
        )
        assert kwargs.get("target_variable") is None, (
            "When generating new unlabeled data, you cannot specify a target variable. The unlabeled data will be "
            "generated using the input_variable."
        )

        super().__init__(
            input_variables=input_variables,
            target_variable=None,
            task_description=task_description,
            **kwargs,
        )
