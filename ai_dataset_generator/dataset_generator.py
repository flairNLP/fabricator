import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union, Tuple

from datasets import Dataset
from haystack.nodes import PromptNode
from haystack.nodes import PromptTemplate as HaystackPromptTemplate
from loguru import logger
from tqdm import tqdm

from ai_dataset_generator.prompts.base import LLMPrompt
from ai_dataset_generator.utils import log_dir, create_timestamp_path, \
    infer_dummy_example


class DatasetGenerator:
    """The DatasetGenerator class is the main class of the ai_dataset_generator package.
    It generates datasets based on a prompt template. The main function is generate()."""

    def __init__(self, prompt_node: PromptNode, max_tries: int = 10):
        """Initialize the DatasetGenerator with a prompt node.

        Args:
            prompt_node (PromptNode): Prompt node / LLM from haystack.
        """
        self.prompt_node = prompt_node
        self._base_log_dir = log_dir()
        self._max_tries = max_tries

    def _setup_log(self, prompt_template: LLMPrompt) -> Path:
        """For every generation run create a new log file.
        Current format: <timestamp>_<prompt_template_name>.jsonl

        Args:
            prompt_template (LLMPrompt): Prompt template to generate the dataset with.

        Returns:
            Path: Path to the log file.

        """
        timestamp_path = create_timestamp_path(self._base_log_dir)
        log_file = Path(f"{timestamp_path}_{prompt_template.__class__.__name__}.jsonl")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch()
        return log_file

    def generate(
        self,
        support_examples: Dataset,
        prompt_template: LLMPrompt,
        unlabeled_examples: Optional[Dataset] = None,
        support_examples_per_prompt: int = 2,
        num_samples_to_generate: int = 10,
        max_prompt_calls: int = 10,
        return_original_dataset: bool = False,
        dry_run: bool = False,
        timeout_per_prompt: Optional[int] = None,
    ) -> Union[Dataset, Tuple[Dataset, Dataset]]:
        """Generate a dataset based on a prompt template and support examples.
        Optionally, unlabeled examples can be provided to annotate unlabeled data.

        Args:
            support_examples (Dataset): Support examples to generate the dataset from.
            prompt_template (LLMPrompt): Prompt template to generate the dataset with.
            unlabeled_examples (Optional[Dataset], optional): Unlabeled examples to annotate. Defaults to None.
            support_examples_per_prompt (int, optional): Number of support examples per prompt. Defaults to 2.
            num_samples_to_generate (int, optional): Number of samples to generate. Defaults to 10.
            max_prompt_calls (int, optional): Maximum number of prompt calls. Defaults to 10.
            return_original_dataset (bool, optional): Whether to return the original dataset. Defaults to False.
            dry_run (bool, optional): Whether to actually generate the examples or just return dummy examples.
            timeout_per_prompt (Optional[int], optional): Timeout per prompt call. Defaults to None.

        Returns:
            Union[Dataset, Tuple[Dataset, Dataset]]: Generated dataset or tuple of generated dataset and original
            dataset.
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
            input_examples = max(max_prompt_calls, num_samples_to_generate) * [
                {prompt_template.input_variables[0]: ""}
            ]
        else:
            input_examples = unlabeled_examples

        generated_dataset, original_dataset = self._inner_generate_loop(
            input_examples,  # type: ignore
            support_examples,
            support_examples_per_prompt,
            prompt_template,
            max_prompt_calls,
            num_samples_to_generate,
            return_original_dataset,
            dry_run,
            timeout_per_prompt
        )

        if return_original_dataset:
            return generated_dataset, original_dataset

        return generated_dataset

    def _try_generate(
        self, prompt_text: str, prompt_template: LLMPrompt, invocation_context: Dict, dry_run: bool
    ) -> Optional[str]:
        """Tries to generate a single example. Restrict the time spent on this.

        Args:
            prompt_text: Prompt text to generate an example for.
            invocation_context: Invocation context to generate an example for.
            dry_run: Whether to actually generate the example or just return a dummy example.

        Returns:
            Generated example
        """

        if dry_run:
            return infer_dummy_example(prompt_template)

        # Haystack internally uses timeouts and retries, so we dont have to do it
        # We dont catch authentification errors here, because we want to fail fast
        prediction = self.prompt_node.run(
            prompt_template=HaystackPromptTemplate(prompt=prompt_text),
            invocation_context=invocation_context,
        )[0]["results"]

        return prediction

    def _inner_generate_loop(
        self,
        input_examples: Iterable[Dict],
        support_examples: Dataset,
        support_examples_per_prompt: int,
        prompt_template: LLMPrompt,
        max_prompt_calls: int,
        num_samples_to_generate: int,
        return_original_dataset: bool,
        dry_run: bool,
        timeout_per_prompt: Optional[int],
    ):
        current_tries_left = self._max_tries
        current_log_file = self._setup_log(prompt_template)

        generated_dataset = defaultdict(list)
        original_dataset = defaultdict(list)

        for prompt_call_idx, input_example in tqdm(
            enumerate(input_examples, start=1), desc="Generating dataset", total=len(input_examples)
        ):
            sampled_support_indices = random.sample(range(len(support_examples)), support_examples_per_prompt)
            sampled_support_examples = support_examples.select(sampled_support_indices)

            prompt_text = prompt_template.get_prompt_text(sampled_support_examples)
            invocation_context = prompt_template.filter_example_by_columns(
                input_example, prompt_template.input_variables
            )

            prediction = self._try_generate(prompt_text, prompt_template, invocation_context, dry_run)

            if prediction is None:
                current_tries_left -= 1
                logger.warning(f"Could not generate example for prompt {prompt_text}.")
                if current_tries_left == 0:
                    logger.warning(
                        f"Max tries ({self._max_tries}) exceeded. Returning generated dataset with"
                        " {len(generated_dataset)} examples."
                    )
                    break

            if len(prediction) == 1:
                prediction = prediction[0]

            # If we have a target variable, we re-use the relevant columns of the input example
            # and add the prediction to the generated dataset
            if prompt_template.target_variable is not None:
                generated_sample = prompt_template.filter_example_by_columns(
                    input_example, prompt_template.input_variables
                )

                for key, value in generated_sample.items():
                    generated_dataset[key].append(value)

                # Try to safely convert the prediction to the type of the target variable
                if prompt_template.target_variable in input_example:
                    prediction = self._convert_prediction(
                        prediction, type(input_example[prompt_template.target_variable])
                    )

                generated_dataset[prompt_template.target_variable].append(prediction)

            else:
                generated_dataset[prompt_template.input_variables[0]].append(prediction)

            log_entry = {
                "prompt": prompt_text,
                "invocation_context": invocation_context,
                "prediction": prediction,
                "target": prompt_template.target_variable
                if prompt_template.target_variable is not None
                else prompt_template.input_variables[0],
            }
            with open(current_log_file, "a", encoding="utf-8") as log_file:
                log_file.write(f"{json.dumps(log_entry)}\n")

            if return_original_dataset:
                for key, value in input_example.items():
                    original_dataset[key].append(value)

            if prompt_call_idx >= max_prompt_calls:
                logger.info("Reached maximum number of prompt calls ({}).", max_prompt_calls)
                break

            if len(generated_dataset) >= num_samples_to_generate:
                logger.info("Generated {} samples.", num_samples_to_generate)
                break

            if timeout_per_prompt is not None:
                time.sleep(timeout_per_prompt)

        generated_dataset = Dataset.from_dict(generated_dataset)

        if return_original_dataset:
            original_dataset = Dataset.from_dict(original_dataset)
            return generated_dataset, original_dataset

        return generated_dataset, None

    def _convert_prediction(self, prediction: str, target_type: type) -> Any:
        """Converts a prediction to the target type.

        Args:
            prediction: Prediction to convert.
            target_type: Type to convert the prediction to.

        Returns:
            Converted prediction.
        """

        if isinstance(prediction, target_type):
            return prediction

        try:
            return target_type(prediction)
        except ValueError:
            logger.warning(
                "Could not convert prediction {} to type {}. "
                "Returning original prediction.", repr(prediction), target_type
            )
            return prediction
