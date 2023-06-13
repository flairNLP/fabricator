from typing import Dict, List

from ai_dataset_generator.prompt_templates.base import TaskPrompt
from langchain.prompts import PromptTemplate, FewShotPromptTemplate


class QuestionGenerationPrompt(TaskPrompt):
    def __init__(self):
        super().__init__(
            fewshot_variables=["context", "question"],
            input_variable=["input_context"],
            task_type="question_generation"
        )

    def get_prompt(self, support_examples: List[Dict[str, str]]):
        example_formatter_template = """Text: {context}\nQuestion: {question}"""

        example_prompt = PromptTemplate(
            input_variables=self.fewshot_variables,
            template=example_formatter_template,
        )

        return FewShotPromptTemplate(
            examples=support_examples,
            prefix="Given a text, create a difficult question that can be answered using the text. The question must describe the context of the text.",
            example_prompt=example_prompt,
            suffix="Text: {input_context}\nQuestion: ",
            input_variables=["input_context"],
        )


class AnswerGenerationPrompt(TaskPrompt):
    def __init__(self):
        super().__init__(
            fewshot_variables=["context", "question", "answer"],
            input_variable=["input_context", "input_question"],
            task_type="answer_generation"
        )

    def get_prompt(self, support_examples: Dict[str, str]):
        example_formatter_template = """Text: {context}\nQuestion: {question}\nAnswer: {answer}"""

        example_prompt = PromptTemplate(
            input_variables=self.fewshot_variables,
            template=example_formatter_template,
        )

        return FewShotPromptTemplate(
            examples=support_examples,
            prefix="Given a text and a question, extract the answer to this question from the text. The answer must be word for word exactly as it appears in the text.",
            example_prompt=example_prompt,
            suffix="Text: {input_context}\nQuestion{input_question}\nAnswer: ",
            input_variables=["input_context", "input_question"],
        )

