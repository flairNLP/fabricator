from ai_dataset_generator.prompt_templates import AnnotationPrompt


class QuestionAnnotationPrompt(AnnotationPrompt):
    def __init__(self):
        super().__init__(
            task_description="Given a text, create a difficult question that can be answered using the text. The question must describe the context of the text.",
            support_set_variables=["context", "question"],
            support_set_formatting_template="""Text: {context}\nQuestion: {question}""",
            annotation_variables=["context"],
            annotation_formatting_template="""Text: {context}\nQuestion: """,
        )


class AnswerAnnotationPrompt(AnnotationPrompt):
    def __init__(self):
        super().__init__(
            task_description="Given a text and a question, extract the answer to this question from the text. The answer must be word for word exactly as it appears in the text.",
            support_set_variables=["context", "question", "answer"],
            support_set_formatting_template="""Text: {context}\nQuestion: {question}\nAnswer: {answer}""",
            annotation_variables=["context", "question"],
            annotation_formatting_template="""Text: {context}\nQuestion: {question}\nAnswer: """,
        )


class ContextAnnotationPrompt(AnnotationPrompt):
    def __init__(self):
        super().__init__(
            task_description="Given a question and a answer, generate a text to which the question and answer fits.",
            support_set_variables=["context", "question", "answer"],
            support_set_formatting_template="""Question: {question}\nAnswer: {answer}\nText: {context}""",
            annotation_variables=["question", "answer"],
            annotation_formatting_template="""Question: {question}\nAnswer: {answer}\nText: """,
        )
