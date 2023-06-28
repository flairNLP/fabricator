import logging
import random
from abc import abstractmethod
from typing import Optional, Union, Tuple

from datasets import Dataset
from haystack.nodes import PromptNode
from haystack.nodes import PromptTemplate as HaystackPromptTemplate

from ai_dataset_generator.prompt_templates import RefactoredBasePrompt

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, prompt_node: PromptNode):
        self.prompt_node = prompt_node

    def generate(
            self,
            support_examples: Dataset,
            prompt_template: RefactoredBasePrompt,
            unlabeled_examples: Optional[Dataset] = None,
            support_examples_per_prompt: int = 2,
            num_samples_to_generate: int = 10,
            max_prompt_calls: int = 10,
            return_original_dataset: bool = False,
    ) -> Union[Dataset, Tuple[Dataset, Dataset]]:
        # Check if required variables of the prompt template occure in every data point
        assert all(field in support_examples.column_names for field in prompt_template.variables_for_examples), \
            "Not all required variables of the prompt template occur in the support examples."

        generated_dataset = {}
        original_dataset = {}

        if unlabeled_examples is None:
            input_examples = iter(max(max_prompt_calls, num_samples_to_generate) * [{prompt_template.input_variables[0]: ""}])
        else:
            input_examples = iter(unlabeled_examples)

        for prompt_call_idx, input_example in enumerate(input_examples, start=1):
            # TODO @All we need ideally one example per class, don't do it purely
            #  randomly, at least for text classification, but I guess also for NER etc
            #  it would be more efficient to have some logic to select the examples
            #  (while of course keeping as much randomness as possible)
            sampled_support_indices = random.sample(range(len(support_examples)), support_examples_per_prompt)
            sampled_support_examples = support_examples.select(sampled_support_indices)

            prompt_text = prompt_template.get_prompt_text(sampled_support_examples)
            invocation_context = prompt_template.filter_example_by_columns(input_example,
                                                                           prompt_template.input_variables)

            pred = self.prompt_node.run(
                prompt_template=HaystackPromptTemplate(name="prompt_text", prompt_text=prompt_text),
                invocation_context=invocation_context,
            )[0]["results"]

            if len(pred) == 1:
                pred = pred[0]

            if prompt_template.target_variable is not None:
                generated_sample = prompt_template.filter_example_by_columns(input_example, prompt_template.input_variables)
                for key, value in generated_sample.items():
                    generated_dataset.setdefault(key, []).append(value)

                if type(pred) is not type(input_example[prompt_template.target_variable]):
                    try:
                        pred = type(input_example[prompt_template.target_variable])(pred)
                    except TypeError:
                        continue
                generated_dataset.setdefault(prompt_template.target_variable, []).append(pred)
            else:
                generated_dataset.setdefault(prompt_template.input_variables[0], []).append(pred)

            if return_original_dataset:
                for key, value in input_example.items():
                    original_dataset.setdefault(key, []).append(value)

            if prompt_call_idx >= max_prompt_calls:
                logger.info(f"Reached maximum number of prompt calls ({max_prompt_calls}).")
                break

            if len(generated_dataset) >= num_samples_to_generate:
                logger.info(f"Generated {num_samples_to_generate} samples.")
                break

        generated_dataset = Dataset.from_dict(generated_dataset)
        original_dataset = Dataset.from_dict(original_dataset)

        if return_original_dataset:
            return generated_dataset, original_dataset
        else:
            return generated_dataset
