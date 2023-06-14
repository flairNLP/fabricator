from ai_dataset_generator.prompt_templates import GenerationPrompt


class TextGenerationPrompt(GenerationPrompt):
    def __init__(self):
        super().__init__(
            task_description="Given a list of text, generate a similar text.",
            support_set_variables=["text"],
            support_set_formatting_template="""Text: {text}""",
            annotation_formatting_template="""Text: """,
        )
