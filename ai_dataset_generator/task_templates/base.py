from typing import Any, Dict


class BaseDataPoint:
    """
    Base class for all task data points. This class is task-agnostic and only contains the most basic information about
    a data point. It is used to ensure a uniform data structure for all task-specific prompt template.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    @property
    def attributes(self) -> Dict[str, Any]:
        """
        Exposes the relevant attributes of a data point in a dict format.
        The key is the attribute name with its corresponding value and the
        dict value.
        """
        raise NotImplementedError
