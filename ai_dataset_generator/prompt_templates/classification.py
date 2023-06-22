from typing import Optional, List, Union

from ai_dataset_generator.prompt_templates import AnnotationPrompt


class LabelClassificationPrompt(AnnotationPrompt):
    def __init__(
        self,
        labels: List[Union[str, int]],
        task_description: Optional[str],
        support_set_variables=["text", "label"],
        support_set_formatting_template="""Text: {text}\nLabel: {label}""",
        annotation_variables=["text"],
        annotation_formatting_template="""Text: {text}\nLabel: """,
    ):
        super().__init__(
            task_description=task_description,
            support_set_variables=support_set_variables,
            support_set_formatting_template=support_set_formatting_template,
            annotation_variables=annotation_variables,
            annotation_formatting_template=annotation_formatting_template,
        )

        # perform some basic checks regarding the labels, e.g., that they not contain
        # spaces or commas, which we use for formatting
        for label in labels:
            if isinstance(label, str):
                assert " " not in label, f"Labels must not contain spaces ({label})"
                assert "," not in label, f"Labels must not contain commas ({label})"
            elif isinstance(label, int):
                pass
            else:
                raise ValueError(f"Labels must be either strings or integers (" f"{type(label)})")

    @staticmethod
    def formatted_labels(labels):
        return ", ".join([str(label) for label in labels])


class AddSingleLabelAnnotationPrompt(LabelClassificationPrompt):
    """
    Prompt template for the following task: Given some text, get the single correct
    label for this text. The LLM is given a text and a list of labels and some
    support examples.
    """

    def __init__(self, labels: List[Union[str, int]], **kwargs):
        super().__init__(
            task_description=f"Given a text, determine the single correct label. It "
            f"must be one of the following labels: "
            f"{self.formatted_labels(labels)}",
            labels=labels,
            **kwargs,
        )


class AddMultiLabelAnnotationPrompt(LabelClassificationPrompt):
    """
    Prompt template for the following task: Given some text, get zero or more correct
    labels for this text. The LLM is given a text and a list of labels and some
    support examples.
    """

    def __init__(self, labels: List[Union[str, int]], **kwargs):
        super().__init__(
            task_description=f"Given a text, determine its correct labels. The text has"
                             f" zero, one, or more correct labels. The labels must be "
                             f"from the following labels: "
                             f"{self.formatted_labels(labels)}",
            labels=labels,
            **kwargs,
        )

