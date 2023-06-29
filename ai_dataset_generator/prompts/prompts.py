from typing import List, Dict, Union, Optional
from datasets import Dataset

from langchain.prompts import PromptTemplate

ClassificationLabels = Union[Dict[str, str], Dict[int, str]]


class DataGenerationPrompt:
    def __init__(
        self,
        input_variables: Union[List[str], str],
        output_format: str = "text",
        example_separator: str = "\n",
        target_variable: Optional[str] = None,
        task_description: Optional[str] = None,
        examples_formatting_template: Optional[str] = None,
        target_formatting_template: Optional[str] = None,
        classification_labels: Optional[ClassificationLabels] = None,
    ):
        # Check output format is valid
        if not output_format in [
            "text",
            "single_label_classification",
            "multi_label_classification",
            "token_classification",
        ]:
            raise ValueError(
                f'Format "{output_format}" is not supported. Choose one of "text", "single_label", "multi_label", "token_classification".'
            )

        # Check, if output format is for classification tasks, that required target_variable and classification_label_options are set
        if target_variable is None and output_format != "text":
            raise ValueError(f"When creating unlabeled data, you cannot use any other output_format than text.")

        # Check if creating unlabeled data, that only one input_variable is passed
        if target_variable is None and output_format == "text":
            assert len(input_variables) == 1, "When creating unlabeled data, you can only use one input_variable."

        # Check, if output format is for classification tasks, that required classification_label_options are set
        if "classification" in output_format and classification_labels is None:
            raise ValueError(
                f"When using output format for classification, you need to specify classification_labels."
            )

        # If only one input_variable is passed, convert it to a list
        if isinstance(input_variables, str):
            input_variables = [input_variables]
        self.input_variables = input_variables
        self.target_variable = target_variable

        if "classification" in output_format:
            formatted_classification_labels = ", ".join([f"{k}: {v}" for k, v in classification_labels.items()])
            if output_format == "single_label_classification":
                classification_suffix = (
                    f" Your prediction must be exactly one of the following labels: {formatted_classification_labels}."
                )
            elif output_format == "multi_label_classification":
                classification_suffix = f" Your prediction must be zero or more of the following labels: {formatted_classification_labels}."
            elif output_format == "token_classification":
                classification_suffix = f" Your prediction must be a list of labels, one for each token. Each label must be one of the following labels: {formatted_classification_labels}."
        else:
            classification_suffix = ""
        self.task_description = task_description + classification_suffix

        # If target_variable is passed, convert it to a list
        # variables_for_examples are the column names for the support set examples
        if target_variable is not None:
            self.variables_for_examples = input_variables + [target_variable]
        else:
            self.variables_for_examples = input_variables

        self.example_separator = example_separator
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
            # Check if we generate unlabeled data, we checked above that only one input_variable is passed
            if target_variable is None and output_format == "text":
                prediction_suffix = ""
            else:
                prediction_suffix = f"{target_variable.capitalize()}: "
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
