from typing import List, Dict

from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
)

from ai_dataset_generator.task_templates.base import BaseDataPoint


class BasePrompt:
    def __init__(
        self,
        task_description: str,
        support_set_variables: List[str],
        support_set_formatting_template: str,
        annotation_variables: List[str],
        annotation_formatting_template: str,
        target_variable: List[str],
    ):
        self.task_description = task_description
        self.support_set_variables = support_set_variables
        self.support_set_formatting_template = support_set_formatting_template
        self.annotation_variables = annotation_variables
        self.annotation_formatting_template = annotation_formatting_template
        self.target_variable = target_variable

    def get_support_template(self):
        return PromptTemplate(
            input_variables=self.support_set_variables,
            template=self.support_set_formatting_template,
        )

    def format_support_examples(self, support_examples: List[BaseDataPoint]) -> List[Dict[str, str]]:
        return [
            {attr: value for attr, value in vars(support_example).items() if attr in self.support_set_variables}
            for support_example in support_examples
        ]

    def format_variables(self, variable: str, value: str) -> Dict[str, str]:
        return {variable: value}

    def format_input(self, input_example: BaseDataPoint) -> Dict[str, str]:
        formatted_inputs = {}

        for attr, value in input_example.attributes.items():
            if attr in self.annotation_variables:
                formatted_input_variable = self.format_variables(attr, value)
                assert len(formatted_input_variable) == 1, "A unformatted input can only return one formatted input"
                formatted_inputs.update(formatted_input_variable)

        return formatted_inputs

    def add_annotation_to_input(self, input_example: BaseDataPoint, prediction: str) -> BaseDataPoint:
        cls = type(input_example)
        constructor_args = {
            attr: value for attr, value in input_example.attributes.items() if attr in self.annotation_variables
        }
        constructor_args[self.target_variable[0]] = prediction.strip()
        return cls(**constructor_args)

    def get_prompt(self, support_examples: List[BaseDataPoint]) -> FewShotPromptTemplate:
        return FewShotPromptTemplate(
            examples=self.format_support_examples(support_examples),
            prefix=self.task_description,
            example_prompt=self.get_support_template(),
            suffix=self.annotation_formatting_template,
            input_variables=self.annotation_variables,
        )


class AnnotationPrompt(BasePrompt):
    def __init__(
        self,
        task_description: str,
        support_set_variables: List[str],
        support_set_formatting_template: str,
        annotation_variables: List[str],
        annotation_formatting_template: str,
    ):
        target_variable = list(set(support_set_variables) - set(annotation_variables))
        if not len(target_variable) == 1:
            raise RuntimeError(
                f"Exactly one target variable must be specified. "
                f"The difference between the support set variables {support_set_variables} and "
                f"the annotation variables {annotation_variables} must be exactly one variable."
            )

        super().__init__(
            task_description=task_description,
            support_set_variables=support_set_variables,
            support_set_formatting_template=support_set_formatting_template,
            annotation_variables=annotation_variables,
            annotation_formatting_template=annotation_formatting_template,
            target_variable=target_variable,
        )


class GenerationPrompt(BasePrompt):
    def __init__(
        self,
        task_description: str,
        support_set_variables: List[str],
        support_set_formatting_template: str,
        annotation_formatting_template: str,
    ):
        if not len(support_set_variables) == 1:
            raise RuntimeError(f"You cannot specify more than one variable for a generation task. ")

        super().__init__(
            task_description=task_description,
            support_set_variables=support_set_variables,
            support_set_formatting_template=support_set_formatting_template,
            annotation_variables=support_set_variables,
            annotation_formatting_template=annotation_formatting_template.strip()
            + f" {{{ support_set_variables[0] }}}",
            target_variable=support_set_variables,
        )
