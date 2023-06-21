import logging
from typing import Any, Optional, List, Union, Dict

from datasets import Dataset
from ai_dataset_generator.task_templates.base import BaseDataPoint

logger = logging.getLogger(__name__)


class SequenceLabelDataPoint(BaseDataPoint):
    """
    A data point for a sequence labeling task.

    Args:
        tokens: A list of tokens
        annotations: List of annotations by ID
        tags: Set of tags corresponding to the specific sequence labeling task
    """

    def __init__(
        self, tokens: List[str], annotations: Optional[Union[List[int], str]], tags: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        self.tokens = tokens
        self.tags = tags
        # Try to convert from string
        if isinstance(annotations, str):
            try:
                self.annotations = eval(annotations)
            except:
                logger.info(f"Could not convert string of annotations to list of annotations: {annotations}")
                self.annotations = annotations
        else:
            self.annotations = annotations

    @classmethod
    def build_data_points_from_dataset(cls, dataset: Dataset, tags_key: str) -> List["SequenceLabelDataPoint"]:
        if "train" in dataset:
            features = dataset["train"].features
        else:
            features = dataset.features

        feature_tags = features[tags_key].feature.names
        tags = {tag: idx for idx, tag in enumerate(list(feature_tags))}

        return [cls(tokens=sample["tokens"], annotations=sample[tags_key], tags=tags) for sample in dataset]

    @property
    def attributes(self) -> Dict[str, Any]:
        return {"tokens": self.tokens, "annotations": self.annotations}
