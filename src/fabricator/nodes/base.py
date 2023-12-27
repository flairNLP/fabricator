from abc import ABC, abstractmethod

from typing import Optional, Dict, Union

from haystack.nodes import PromptNode as HaystackPromptNode
from haystack.nodes.prompt import PromptTemplate as HaystackPromptTemplate


class Node(ABC):
    @abstractmethod
    def run(self, prompt_template):
        pass


class PromptNode(Node):
    def __init__(self, model_name_or_path: str, *args, **kwargs) -> None:
        self._prompt_node = HaystackPromptNode(model_name_or_path, *args, **kwargs)

    def run(
        self,
        prompt_template: Optional[Union[str, HaystackPromptTemplate]],
        invocation_context: Optional[Dict[str, any]] = None,
    ):
        return self._prompt_node.run(prompt_template, invocation_context=invocation_context)[0]["results"]
