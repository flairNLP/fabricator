"""
Prompting Templates for Sequence Labeling.

The tagging scheme follows CoNNL-2003 (https://aclanthology.org/W03-0419/) standard and
the commonly used Huggingface tag representation.

"""

from typing import Dict, Optional

from ai_dataset_generator.prompt_templates import AnnotationPrompt


class SequenceLabelingPrompt(AnnotationPrompt):
    def __init__(
        self,
        tags: Dict[str, int],
        task_description: Optional[str],
        support_set_variables=["tokens", "annotations"],
        support_set_formatting_template="""Text: {tokens}\nAnnotations: {annotations}""",
        annotation_variables=["tokens"],
        annotation_formatting_template="""Text: {tokens}\nAnnotations: """,
    ):
        self.tags = tags
        if task_description is None:
            task_description = (
                f"Given a list of tokens, generate a list of annotations for every token with according to their label based on following tags: {self.formatted_tags(tags)}",
            )
        super().__init__(
            task_description=task_description,
            support_set_variables=support_set_variables,
            support_set_formatting_template=support_set_formatting_template,
            annotation_variables=annotation_variables,
            annotation_formatting_template=annotation_formatting_template,
        )

    def formatted_tags(self, tags):
        return ", ".join([f"{k}: {v}" for k, v in tags.items()])

    def format_variables(self, variable: str, value: str) -> Dict[str, str]:
        return {variable: value}


class NamedEntityAnnotationPrompt(SequenceLabelingPrompt):
    def __init__(self, tags: Dict[str, int], **kwargs):
        super().__init__(
            tags,
            task_description=f"Given a list of tokens, generate a list of annotations for every token with according to their entity based on following NER tags: {self.formatted_tags(tags)}",
            **kwargs,
        )


class PartOfSpeechAnnotationPrompt(SequenceLabelingPrompt):
    def __init__(self, tags: Dict[str, int], **kwargs):
        super().__init__(
            tags,
            task_description=f"Given a list of tokens, generate a list of annotations for every token with according to their entity based on following Part-of-Speech tags: {self.formatted_tags(tags)}",
            **kwargs,
        )


class ChunkingAnnotationPrompt(SequenceLabelingPrompt):
    def __init__(self, tags: Dict[str, int], **kwargs):
        super().__init__(
            tags,
            task_description=f"Given a list of tokens, generate a list of annotations for every token with according to their entity based on following Chunking tags: {self.formatted_tags(tags)}",
            **kwargs,
        )
