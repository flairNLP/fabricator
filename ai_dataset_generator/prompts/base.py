import logging
from typing import List, Dict, Union, Optional
from datasets import Dataset

from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

ClassificationLabels = List[str]
ID2Label = Dict[int, str]
ClassificationOptions = Union[ClassificationLabels, ID2Label]


class LLMPrompt:
    """Base class for prompt generation. This class formats the prompt for the fewshot / support set examples
    and the target variable such that the dataset generator can simply put in the invocation context."""

    def __init__(
        self,
        input_variables: Union[List[str], str],
        target_variable: Optional[str] = None,
        task_description: Optional[str] = None,
        label_options: Optional[ClassificationOptions] = None,
        examples_formatting_template: Optional[str] = None,
        target_formatting_template: Optional[str] = None,
        example_separator: str = "\n\n",
        inner_example_separator: str = "\n",
    ):
        """Base class for prompt generation. This class formats the prompt for the fewshot / support set examples.

        Args:
            input_variables (Union[List[str], str]): List or string of input variables / column names for the
            fewshot / support set examples
            target_variable (Optional[str], optional): Target variable / column name. The column is annotated by the
            LLM. Defaults to None.
            task_description (Optional[str], optional): Task description for the prompt (prefix). Defaults to None.
            label_options (Optional[ClassificationOptions], optional): Label options for the LLM to choose from.
            Defaults to None.
            examples_formatting_template (Optional[str], optional): Template for formatting the fewshot / support set
            examples. Defaults to None.
            target_formatting_template (Optional[str], optional): Template for formatting the target variable.
            Defaults to None.
            example_separator (str, optional): Separator between the fewshot / support set examples.
            Defaults to "\n\n".
            inner_example_separator (str, optional): Separator in-between a single fewshot examples. Defaults to "\n".

        Raises:
            AttributeError: If label_options is not a dict or list
            KeyError: If the task_description cannot be formatted with the variable 'label_options'

        Examples:
            >>> from datasets import load_dataset
            >>> from ai_dataset_generator.prompts import ClassLabelPrompt
            >>> input_variable = "text"
            >>> target_variable = "coarse_label"
            >>>
            >>> dataset = load_dataset("trec", split="train")
            >>> id2label = {k: v for k, v in enumerate(dataset.features[target_variable].names)}
            >>> fewshot_examples = dataset.select([1,2,3])
            >>>
            >>> prompt = ClassLabelPrompt(
            >>>    input_variables=input_variable,
            >>>    target_variable=target_variable,
            >>>    label_options=id2label,
            >>> )
            >>>
            >>> raw_prompt = prompt.get_prompt_text(fewshot_examples)
            >>> print(raw_prompt)
        """
        if label_options is not None:
            if "{label_options}" not in task_description:
                logger.warning(
                    "{label_options} is not found in the task_description. If you want to limit your answers to "
                    "these information, make sure to include {label_options}."
                )

            if isinstance(label_options, dict):
                formatted_label_options = ", ".join([f"{k}: {v}" for k, v in label_options.items()])
            elif isinstance(label_options, list):
                formatted_label_options = ", ".join(label_options)
            else:
                raise AttributeError("Type of label options must be either Dict[int, str] or List[str]")

            try:
                task_description = task_description.format(label_options=formatted_label_options)
            except KeyError as exc:
                raise KeyError(
                    "The provided task description cannot be formatted with the variable 'label_options'. "
                    "You need to include {label_options} in the task_description string to limit your prompt output "
                    "to these labels. For example: task_description='[...] limit your answer to: {label_options}.'"
                ) from exc

        # If only one input_variable is passed, convert it to a list
        if isinstance(input_variables, str):
            input_variables = [input_variables]
        self.input_variables = input_variables
        self.target_variable = target_variable
        self.task_description = task_description
        self.example_separator = example_separator
        self.inner_example_separator = inner_example_separator

        # variables_for_examples are the column names for the fewshot / support set examples
        if target_variable is not None:
            self.variables_for_examples = input_variables + [target_variable]
        else:
            self.variables_for_examples = input_variables

        # Create prompt template for support examples
        if examples_formatting_template is None:
            examples_formatting_template = self.inner_example_separator.join(
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
        """Filter single example by columns.

        Args:
            example (Dict[str, str]): Example to filter
            columns (List[str]): Columns to keep

        Returns:
            Dict[str, str]: Filtered example
        """
        filtered_example = {key: value for key, value in example.items() if key in columns}
        return filtered_example

    def filter_examples_by_columns(self, dataset: Dataset, columns: List[str]) -> List[Dict[str, str]]:
        """Filter examples by columns.

        Args:
            dataset (Dataset): Dataset to filter
            columns (List[str]): Columns to keep

        Returns:
            List[Dict[str, str]]: Filtered examples
        """
        filtered_inputs = []
        for example in dataset:
            filtered_inputs.append(self.filter_example_by_columns(example, columns))
        return filtered_inputs

    def get_prompt_text(self, examples: Dataset) -> str:
        """Get prompt text for the given examples.

        Args:
            examples (Dataset): Examples to use for the prompt

        Returns:
            str: Prompt text
        """
        examples = self.filter_examples_by_columns(examples, self.variables_for_examples)
        formatted_examples = [self.example_prompt.format_prompt(**example).text for example in examples]
        prompt_text = self.example_separator.join(
            [self.task_description] + formatted_examples + [self.target_formatting_template]
        )
        return prompt_text
