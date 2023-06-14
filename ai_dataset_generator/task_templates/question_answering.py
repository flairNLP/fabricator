import logging
from typing import Optional

from ai_dataset_generator.task_templates.base import BaseDataPoint

logger = logging.getLogger(__name__)


class ExtractiveQADataPoint(BaseDataPoint):
    """
    A data point for a question answering task.

    Args:
        title: Title of the document.
        question: Question to be answered.
        context: Context in which the question should be answered.
        is_impossible: Whether the question can be answered from the context.
        is_extractive: Whether the answer is extractive (i.e. a span of text from the context)
        answer: Answer to the question.
        answer_start: Start position of the answer in the context.
    """

    def __init__(
        self,
        context: str,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        title: Optional[str] = None,
        answer_start: Optional[int] = None,
    ):
        super().__init__()
        self.context = context
        self.question = question
        self.answer = answer
        self.title = title
        if answer is not None and answer_start is None:
            self.calculate_answer_start()

    def calculate_answer_start(self):
        """
        Calculate the start position of the answer in the context.
        """
        answer_start = self.context.find(self.answer)
        if answer_start < 0:
            logger.warning(
                f'Could not calculate the answer start because the context "{self.context}" '
                f'does not contain the answer "{self.answer}".'
            )
            self.answer_start = -1
        else:
            # check that the answer doesn't occur more than once in the context
            second_answer_start = self.context.find(self.answer, answer_start + 1)
            if second_answer_start >= 0:
                logger.warning(
                    "Could not calculate the answer start because the context contains the answer more than once."
                )
                self.answer_start = -1
            else:
                self.answer_start = answer_start
