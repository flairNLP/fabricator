from typing import Dict
from abc import abstractmethod


class TaskDataPoint:
    """
    Base class for all task data points. This class is task-agnostic and only contains the most basic information about
    a data point. It is used to ensure a uniform data structure for all task-specific prompt template.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
