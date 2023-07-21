from typing import List, Dict, Union, Optional

from datasets import Dataset
from langchain.prompts import PromptTemplate
from loguru import logger

ClassificationLabels = List[str]
ID2Label = Dict[int, str]
ClassificationOptions = Union[ClassificationLabels, ID2Label]


class BasePrompt:
    """Base class for prompt generation. This class formats the prompt for the fewshot / support set examples
    and the target variable such that the dataset generator can simply put in the invocation context."""

    def __init__(
        self,
        task_description: str,
        generate_data_for_column: Optional[str] = None,
        fewshot_example_columns: Optional[Union[List[str], str]] = None,
        label_options: Optional[ClassificationOptions] = None,
        fewshot_formatting_template: Optional[str] = None,
        target_formatting_template: Optional[str] = None,
        fewshot_example_separator: str = "\n\n",
        inner_fewshot_example_separator: str = "\n",
    ):
        """Base class for prompt generation. This class formats the prompt for the fewshot / support set examples.

        Args:
            task_description (Optional[str], optional): Task description for the prompt (prefix).
            generate_data_for_column (Optional[str], optional): The column name to generate data for. Defaults to None.
            fewshot_example_columns (Union[List[str], str]): List of strings or string of column names for the
            fewshot / support set examples. Defaults to None.
            label_options (Optional[ClassificationOptions], optional): Label options for the LLM to choose from.
            Defaults to None.
            fewshot_formatting_template (Optional[str], optional): Template for formatting the fewshot / support set
            examples. Defaults to None.
            target_formatting_template (Optional[str], optional): Template for formatting the target variable.
            Defaults to None.
            fewshot_example_separator (str, optional): Separator between the fewshot / support set examples.
            Defaults to "\n\n".
            inner_fewshot_example_separator (str, optional): Separator in-between a single fewshot examples.
            Defaults to "\n".

        Raises:
            AttributeError: If label_options is not a dict or list
            KeyError: If the task_description cannot be formatted with the variable 'label_options'
            ValueError: You need specify either generate_data_for_column or
                generate_data_for_column + fewshot_example_columns. Only fewshot_example_columns is not supported.
        """
        if self._check_label_options_in_task_description(label_options, task_description):
            task_description = self._format_task_description(label_options, task_description)

        # If only one input_variable is passed, convert it to a list
        self.task_description = task_description
        if isinstance(generate_data_for_column, str) and generate_data_for_column:
            generate_data_for_column = [generate_data_for_column]
        self.generate_data_for_column = generate_data_for_column
        if isinstance(fewshot_example_columns, str) and fewshot_example_columns:
            fewshot_example_columns = [fewshot_example_columns]
        self.fewshot_example_columns = fewshot_example_columns
        self.fewshot_example_separator = fewshot_example_separator
        self.inner_fewshot_example_separator = inner_fewshot_example_separator

        if fewshot_example_columns:
            self.relevant_columns_for_fewshot_examples = self.fewshot_example_columns + self.generate_data_for_column
        elif generate_data_for_column:
            self.relevant_columns_for_fewshot_examples = self.generate_data_for_column
        else:
            self.relevant_columns_for_fewshot_examples = None

        # Create prompt template for fewshot examples
        if self.relevant_columns_for_fewshot_examples:
            if fewshot_formatting_template is None:
                fewshot_formatting_template = self.inner_fewshot_example_separator.join(
                    [f"{var}: {{{var}}}" for var in self.relevant_columns_for_fewshot_examples]
                )

            self.example_prompt = PromptTemplate(
                input_variables=self.relevant_columns_for_fewshot_examples,
                template=fewshot_formatting_template,
            )

        # Create format template for targets
        if target_formatting_template is None:
            self.target_formatting_template = self._infer_target_formatting_template()
        else:
            self.target_formatting_template = target_formatting_template

    @staticmethod
    def _check_label_options_in_task_description(label_options: ClassificationOptions, task_description) -> bool:
        """Check if label_options are included in the task_description.

        Args:
            label_options (ClassificationOptions): Label options for the LLM to choose from.
            task_description (str): Task description for the prompt (prefix).

        Returns:
            bool: True if label_options can be included in the task_description, False otherwise.
        """
        if label_options and "{label_options}" in task_description:
            return True

        logger.info(
            "label_options are None or task_description cannot be formatted. If you want to limit your answers to "
            "these information, make sure to include '{label_options}' into the task_description."
        )
        return False

    @staticmethod
    def _format_task_description(label_options: ClassificationOptions, task_description: str) -> str:
        """Format task description with label options.

        Args:
            label_options (ClassificationOptions): Label options for the LLM to choose from.
            task_description (str): Task description for the prompt (prefix).

        Returns:
            str: Formatted task description.
        """
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
        return task_description

    def _infer_target_formatting_template(self) -> str:
        """Infer target formatting template from input columns and label column.

        Returns:
            str: Target formatting template
        """
        if self.generate_data_for_column and self.fewshot_example_columns:
            target_template = self.inner_fewshot_example_separator.join(
                [f"{var}: {{{var}}}" for var in self.fewshot_example_columns] +
                [f"{self.generate_data_for_column[0]}: "]
            )

        elif self.generate_data_for_column and not self.fewshot_example_columns:
            target_template = f"{self.generate_data_for_column[0]}: "

        elif not self.generate_data_for_column and not self.fewshot_example_columns:
            target_template = "text: "

        else:
            raise ValueError("Either generate_data_for_column or generate_data_for_column + fewshot_example_columns "
                             "must be provided to infer target template.")

        return target_template

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

    def get_prompt_text(self, examples: Optional[Dataset] = None) -> str:
        """Get prompt text for the given examples.

        Args:
            examples (Dataset): Examples to use for the prompt

        Returns:
            str: Prompt text
        """
        if examples is not None:
            examples = self.filter_examples_by_columns(examples, self.relevant_columns_for_fewshot_examples)
            formatted_examples = [self.example_prompt.format_prompt(**example).text for example in examples]
            prompt_text = self.fewshot_example_separator.join(
                [self.task_description] + formatted_examples + [self.target_formatting_template]
            )
        else:
            prompt_text = self.fewshot_example_separator.join(
                [self.task_description] + [self.target_formatting_template]
            )
        return prompt_text
