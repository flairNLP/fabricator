import logging
from typing import Optional, List, Union

from ai_dataset_generator.task_templates.base import BaseDataPoint

logger = logging.getLogger(__name__)


# TODO: Should this all be sublcassed for NER, POS and Chunking?
class SequenceLabelDataPoint(BaseDataPoint):
    """
    A data point for a sequence labeling task.

    Args:
        tokens: A list of tokens
        annotations: List of annotations by ID
    """

    def __init__(self, tokens: List[str], annotations: Optional[Union[List[int], str]]):
        super().__init__()
        self.tokens = tokens
        # Try to convert from string
        if isinstance(annotations, str):
            try:
                self.annotations = eval(annotations)
            except:
                logger.info(f"Could not convert string of annotations to list of annotations: {annotations}")
                self.annotations = annotations
        else:
            self.annotations = annotations
