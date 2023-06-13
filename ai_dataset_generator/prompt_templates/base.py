from typing import List, Dict
from abc import abstractmethod

from langchain.prompts import FewShotPromptTemplate


class TaskPrompt:

    def __init__(self, fewshot_variables: List[str], input_variable: List[str], task_type):
        self.fewshot_variables = fewshot_variables
        self.input_variable = input_variable
        self.task_type = task_type

    @abstractmethod
    def get_prompt(self, support_examples: List[Dict[str, str]]) -> FewShotPromptTemplate:
        raise NotImplementedError
