from typing import Any, Dict, Optional, Union

from ai_dataset_generator.task_templates.base import BaseDataPoint


class SingleLabelClassificationDataPoint(BaseDataPoint):
    """
    A data point for a single label classification task.

    Args:
        text: Text of the document.
        label: Label of the document.
    """

    def __init__(
        self, text: str, label: Optional[Union[int, str]] = None,
    ):
        super().__init__()
        self.text = text
        self.label = label

        if label:
            # perform some basic checks regarding the labels, e.g., that they not
            # contain spaces or commas, which we use for formatting
            if isinstance(label, str):
                assert " " not in label, f"Labels must not contain spaces ({label})"
                assert "," not in label, f"Labels must not contain commas ({label})"
            elif isinstance(label, int):
                pass
            else:
                raise ValueError(f"Labels must be either strings or integers (" f"{type(label)})")

    @property 
    def attributes(self) -> Dict[str, Any]:
        return {"text": self.text, "label": self.label}
