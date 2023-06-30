import logging
import random
from collections import defaultdict
from typing import Dict, Iterable, Optional, Union, Tuple

from datasets import Dataset
from haystack.nodes import PromptNode
from haystack.nodes import PromptTemplate as HaystackPromptTemplate

from ai_dataset_generator.prompts import DataGenerationPrompt
from ai_dataset_generator.utils import log_dir, create_timestamp_path

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, prompt_node: PromptNode):
        self.prompt_node = prompt_node
        self._log_dir = log_dir()

    def _setup_logging(self, prompt_template: DataGenerationPrompt, support_examples: Dataset):
        # Create log dir
        timestamp_path = create_timestamp_path(self._log_dir)
        self._log_dir = timestamp_path

    def generate(
        self,
        support_examples: Dataset,
        prompt_template: DataGenerationPrompt,
        unlabeled_examples: Optional[Dataset] = None,
        support_examples_per_prompt: int = 2,
        num_samples_to_generate: int = 10,
        max_prompt_calls: int = 10,
        return_original_dataset: bool = False,
        dry_run: bool = False,
    ) -> Union[Dataset, Tuple[Dataset, Dataset]]:
        """
        Generates a dataset based on a prompt template and support examples.

        Args:
            support_examples: Examples to support the generation of new examples.
            prompt_template: Prompt template to generate new examples.
            unlabeled_examples: Unlabeled examples to generate new examples for.
            support_examples_per_prompt: Number of support examples to use per prompt.
            num_samples_to_generate: Number of samples to generate per prompt.
            max_prompt_calls: Maximum number of calls to the LLM.
            return_original_dataset: Whether to return the original dataset as well.
            dry_run: Whether to actually generate the dataset or just return a dummy dataset.

        Returns:
            Generated dataset, optionally together with the original dataset.

        """
        # Check if required variables of the prompt template occure in every data point
        assert all(
            field in support_examples.column_names for field in prompt_template.variables_for_examples
        ), "Not all required variables of the prompt template occur in the support examples."

        if unlabeled_examples is None:
            assert (
                len(prompt_template.input_variables) == 1
            ), "When creating unlabeled data, you can only use one input_variable."
            assert isinstance(
                prompt_template.input_variables[0], str
            ), "The input_variable must be a string, indicating the column to generate unlabeled data for."

        if unlabeled_examples is None:
            input_examples = iter(
                max(max_prompt_calls, num_samples_to_generate) * [{prompt_template.input_variables[0]: ""}]
            )
        else:
            input_examples = iter(unlabeled_examples)

        generated_dataset, original_dataset = self._inner_generate_loop(
            input_examples, # type: ignore
            support_examples,
            support_examples_per_prompt,
            prompt_template,
            max_prompt_calls,
            num_samples_to_generate,
            dry_run
        )

        generated_dataset = Dataset.from_dict(generated_dataset)
        original_dataset = Dataset.from_dict(original_dataset)

        if return_original_dataset:
            return generated_dataset, original_dataset
        else:
            return generated_dataset

    def _inner_generate_loop(
        self,
        input_examples: Iterable[Dict],
        support_examples: Dataset,
        support_examples_per_prompt: int,
        prompt_template: DataGenerationPrompt,
        max_prompt_calls: int,
        num_samples_to_generate: int,
        dry_run: bool
    ):
        generated_dataset = defaultdict(list)
        original_dataset = defaultdict(list)

        for prompt_call_idx, input_example in enumerate(input_examples, start=1):
            sampled_support_indices = random.sample(range(len(support_examples)), support_examples_per_prompt)
            sampled_support_examples = support_examples.select(sampled_support_indices)

            prompt_text = prompt_template.get_prompt_text(sampled_support_examples)
            invocation_context = prompt_template.filter_example_by_columns(
                input_example, prompt_template.input_variables
            )

            if dry_run:
                pred = "<Dry Run>"
            else:
                pred = self.prompt_node.run(
                    prompt_template=HaystackPromptTemplate(name="prompt_text", prompt_text=prompt_text),
                    invocation_context=invocation_context,
                )[0]["results"]

            if len(pred) == 1:
                pred = pred[0]

            if prompt_template.target_variable is not None:
                generated_sample = prompt_template.filter_example_by_columns(
                    input_example, prompt_template.input_variables
                )
                for key, value in generated_sample.items():
                    generated_dataset[key].append(value)

                if type(pred) is not type(input_example[prompt_template.target_variable]):
                    try:
                        pred = type(input_example[prompt_template.target_variable])(pred)
                    except TypeError:
                        continue

                generated_dataset[prompt_template.target_variable].append(pred)
            else:
                generated_dataset[prompt_template.input_variables[0]].append(pred)
            
            # As long as we are not dealing with millions of examples, we can safely populate
            for key, value in input_example.items():
                original_dataset[key].append(value)

            if prompt_call_idx >= max_prompt_calls:
                logger.info(f"Reached maximum number of prompt calls ({max_prompt_calls}).")
                break

            if len(generated_dataset) >= num_samples_to_generate:
                logger.info(f"Generated {num_samples_to_generate} samples.")
                break

        return generated_dataset, original_dataset
