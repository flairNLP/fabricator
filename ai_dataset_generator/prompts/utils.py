from datasets import Dataset, QuestionAnsweringExtractive, TextClassification, TaskTemplate
from .base import BasePrompt

DEFAULT_TEXT_CLASSIFICATION = "Classify the following texts exactly into one of the following categories: {" \
                              "label_options}."
DEFAULT_QA = "Given a context and a question, generate an answer that occurs exactly and only once in the text."


def infer_prompt_from_task_template(task_template: TaskTemplate):
    """Infer TextLabelPrompt or ClassLabelPrompt with correct parameters from a task template's metadata."""
    if isinstance(task_template, QuestionAnsweringExtractive):
        return BasePrompt(
            task_description=DEFAULT_QA,
            generate_data_for_column="answer",  # assuming the dataset was preprocessed with preprocess_squad_format
            # otherwise dataset.task_templates[0]["answers_column"]
            fewshot_example_columns=[task_template.context_column, task_template.question_column],
        )
    if isinstance(task_template, TextClassification):
        return BasePrompt(
            task_description=DEFAULT_TEXT_CLASSIFICATION,
            generate_data_for_column=task_template.label_column,
            fewshot_example_columns=task_template.text_column,
            label_options=dict(enumerate(task_template.label_schema["labels"].names)),
        )
    else:
        raise ValueError(
            f"Automatic prompt is only supported for QuestionAnsweringExtractive and "
            f"TextClassification tasks but not for {type(task_template)}. You need to "
            f"specify the prompt manually."
        )


def infer_prompt_from_dataset(dataset: Dataset):
    """Infer TextLabelPrompt or ClassLabelPrompt with correct parameters from a dataset's metadata."""
    if not dataset.task_templates:
        raise ValueError(
            "Dataset must have exactly one task template but there is none. You need to specify the "
            "prompt manually."
        )
    if len(dataset.task_templates) > 1:
        raise ValueError(
            f"Automatic prompt is only supported for datasets with exactly one task template but yours "
            f"has {len(dataset.task_templates)}. You need to specify the prompt manually."
        )
    else:
        return infer_prompt_from_task_template(dataset.task_templates[0])
