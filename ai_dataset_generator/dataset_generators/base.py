import logging
import random
from abc import abstractmethod
from typing import List, Optional

from langchain.llms import BaseLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ai_dataset_generator.task_templates.base import TaskDataPoint
from ai_dataset_generator.prompt_templates.base import TaskPrompt

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    @abstractmethod
    def postprocess_prompt(self, pred: str) -> TaskDataPoint:
        pass

    def generate(
        self,
        unlabeled_examples: List[TaskDataPoint],
        support_examples: List[TaskDataPoint],
        prompt_template: TaskPrompt,
        support_examples_per_prompt: int = 2,
        num_samples_to_generate: int = 10,
        max_prompt_calls: int = 20,
    ):
        if not all(hasattr(obj, attr) for obj in support_examples for attr in prompt_template.fewshot_variables):
            raise RuntimeError("Not all samples have the required attributes for the generation prompt template.")

        generated_samples: List[TaskDataPoint] = []

        for prompt_call_idx, input_example in enumerate(unlabeled_examples, start=1):
            # Sample only X examples for each prompt
            sampled_support_examples_for_prompt = random.sample(support_examples, support_examples_per_prompt)
            # Prompt template class knows relevant fewshot variables to fill prompt with support examples
            prompt_support_examples = [{attr: value for attr, value in vars(sampled_support_example).items() if attr in prompt_template.fewshot_variables} for sampled_support_example in sampled_support_examples_for_prompt]
            # Create fewshot template from langchain
            prompt = prompt_template.get_prompt(prompt_support_examples)
            # create LLMChain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            # Prompt template class knows relevant input variables to fill prompt with unlabeled example
            input = {f"input_{attr}": value for attr, value in vars(input_example).items() if
                     attr in [input_var.replace("input_", "") for input_var in prompt_template.input_variable]}
            # run chain
            pred = chain.run(**input)

            # postprocess tbd, currently only appends string from LLM
            generated_samples.append(pred)

            if prompt_call_idx > max_prompt_calls:
                logger.info(f"Reached maximum number of prompt calls ({max_prompt_calls}).")
                break

            if len(generated_samples) >= num_samples_to_generate:
                logger.info(f"Generated {num_samples_to_generate} samples.")
                break

        return generated_samples
