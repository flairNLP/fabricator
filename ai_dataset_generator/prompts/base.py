import logging
from typing import List, Dict, Union, Optional
from datasets import Dataset

from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class LLMPrompt:
    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: Optional[str] = None,
        task_description: Optional[str] = None,
        examples_formatting_template: Optional[str] = None,
        target_formatting_template: Optional[str] = None,
        example_separator: str = "\n",
    ):
        # If only one input_variable is passed, convert it to a list
        if isinstance(input_variables, str):
            input_variables = [input_variables]
        self.input_variables = input_variables
        self.target_variable = target_variable
        self.task_description = task_description
        self.example_separator = example_separator

        # variables_for_examples are the column names for the fewshot / support set examples
        if target_variable is not None:
            self.variables_for_examples = input_variables + [target_variable]
        else:
            self.variables_for_examples = input_variables

        # Create prompt template for support examples
        if examples_formatting_template is None:
            examples_formatting_template = self.example_separator.join(
                [f"{var.capitalize()}: {{{var}}}" for var in self.variables_for_examples]
            )

        self.example_prompt = PromptTemplate(
            input_variables=self.variables_for_examples,
            template=examples_formatting_template,
        )

        # Create format template for variable annotation or unlabeled generation
        if target_formatting_template is None:
            if target_variable is not None:
                prediction_suffix = f"{target_variable.capitalize()}: "
            else:
                prediction_suffix = ""

            self.target_formatting_template = "\n".join(
                [f"{var.capitalize()}: {{{var}}}" for var in input_variables] + [prediction_suffix]
            )

        else:
            self.target_formatting_template = target_formatting_template

    @staticmethod
    def filter_example_by_columns(example: Dict[str, str], columns: List[str]) -> Dict[str, str]:
        filtered_example = {key: value for key, value in example.items() if key in columns}
        return filtered_example

    def filter_examples_by_columns(self, dataset: Dataset, columns: List[str]) -> List[Dict[str, str]]:
        filtered_inputs = []
        for example in dataset:
            filtered_inputs.append(self.filter_example_by_columns(example, columns))
        return filtered_inputs

    def get_prompt_text(self, examples: Dataset) -> str:
        examples = self.filter_examples_by_columns(examples, self.variables_for_examples)
        formatted_examples = [self.example_prompt.format_prompt(**example).text for example in examples]
        prompt_text = self.example_separator.join(
            [self.task_description] + formatted_examples + [self.target_formatting_template]
        )
        return prompt_text
