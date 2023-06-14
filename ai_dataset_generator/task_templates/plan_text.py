from ai_dataset_generator.task_templates.base import BaseDataPoint


class TextDataPoint(BaseDataPoint):
    """
    A data point for creating unlabeled data.

    Args:
        text: domain-specific text.
    """

    def __init__(self, text: str):
        super().__init__()
        self.text = text
