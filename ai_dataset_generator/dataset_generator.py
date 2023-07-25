import json
import time

from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Optional, Union, Tuple, List
from tqdm import tqdm
from loguru import logger

from datasets import Dataset
from numpy.random import choice
from haystack.nodes import PromptNode
from haystack.nodes import PromptTemplate as HaystackPromptTemplate

from ai_dataset_generator.prompts.base import BasePrompt
from ai_dataset_generator.samplers import single_label_stratified_sample
from ai_dataset_generator.utils import log_dir, create_timestamp_path


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

    def _setup_log(self, prompt_template: BasePrompt) -> Path:
        """For every generation run create a new log file.
        Current format: <timestamp>_<prompt_template_name>.jsonl

        Args:
            prompt_template (BasePrompt): Prompt template to generate the dataset with.

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
        prompt_template: BasePrompt,
        fewshot_dataset: Optional[Dataset] = None,
        fewshot_examples_per_class: int = 1,
        fewshot_label_sampling_strategy: str = None,
        fewshot_sampling_column: str = None,
        unlabeled_dataset: Optional[Dataset] = None,
        return_unlabeled_dataset: bool = False,
        max_prompt_calls: int = 10,
        num_samples_to_generate: int = 10,
        timeout_per_prompt: Optional[int] = None,
    ) -> Union[Dataset, Tuple[Dataset, Dataset]]:
        """Generate a dataset based on a prompt template and support examples.
        Optionally, unlabeled examples can be provided to annotate unlabeled data.

        Args:
            prompt_template (BasePrompt): Prompt template to generate the dataset with.
            fewshot_dataset (Dataset): Support examples to generate the dataset from. Defaults to None.
            fewshot_examples_per_class (int, optional): Number of support examples for a certain class per prompt.
                Defaults to 1.
            fewshot_label_sampling_strategy (str, optional): Sampling strategy for support examples. Defaults to
                None. "uniform" sampling strategy means that one label is sampled and only fewshot examples for that
                class are used. "stratified" sampling strategy means that all classes in support examples are used.
            fewshot_sampling_column (str, optional): Column to sample from. Defaults to None and function will try
                to sample from the generate_data_for_column attribute of the prompt template.
            unlabeled_dataset (Optional[Dataset], optional): Unlabeled examples to annotate. Defaults to None.
            return_unlabeled_dataset (bool, optional): Whether to return the original dataset. Defaults to False.
            max_prompt_calls (int, optional): Maximum number of prompt calls. Defaults to 10.
            num_samples_to_generate (int, optional): Number of samples to generate. Defaults to 10.
            timeout_per_prompt (Optional[int], optional): Timeout per prompt call. Defaults to None.

        Returns:
            Union[Dataset, Tuple[Dataset, Dataset]]: Generated dataset or tuple of generated dataset and original
            dataset.
        """
        if fewshot_dataset:
            self._assert_fewshot_dataset_matches_prompt(prompt_template, fewshot_dataset)

        assert fewshot_label_sampling_strategy in [None, "uniform", "stratified"], \
            "Sampling strategy must be 'uniform' or 'stratified'"

        if fewshot_dataset and not fewshot_sampling_column:
            fewshot_sampling_column = prompt_template.generate_data_for_column[0]

        generated_dataset, original_dataset = self._inner_generate_loop(
            prompt_template,
            fewshot_dataset,
            fewshot_examples_per_class,
            fewshot_label_sampling_strategy,
            fewshot_sampling_column,
            unlabeled_dataset,
            return_unlabeled_dataset,
            max_prompt_calls,
            num_samples_to_generate,
            timeout_per_prompt
        )

        if return_unlabeled_dataset:
            return generated_dataset, original_dataset

        return generated_dataset

    def _try_generate(
        self, prompt_text: str, invocation_context: Dict,
    ) -> Optional[str]:
        """Tries to generate a single example. Restrict the time spent on this.

        Args:
            prompt_text: Prompt text to generate an example for.
            invocation_context: Invocation context to generate an example for.
            dry_run: Whether to actually generate the example or just return a dummy example.

        Returns:
            Generated example
        """

        # Haystack internally uses timeouts and retries, so we dont have to do it
        # We dont catch authentification errors here, because we want to fail fast
        prediction = self.prompt_node.run(
            prompt_template=HaystackPromptTemplate(prompt=prompt_text),
            invocation_context=invocation_context,
        )[0]["results"]

        return prediction

    def _inner_generate_loop(
        self,
        prompt_template: BasePrompt,
        fewshot_dataset: Dataset,
        fewshot_examples_per_class: int,
        fewshot_label_sampling_strategy: str,
        fewshot_sampling_column: str,
        unlabeled_dataset: Dataset,
        return_unlabeled_dataset: bool,
        max_prompt_calls: int,
        num_samples_to_generate: int,
        timeout_per_prompt: Optional[int],
    ):
        current_tries_left = self._max_tries
        current_log_file = self._setup_log(prompt_template)

        generated_dataset = defaultdict(list)
        original_dataset = defaultdict(list)

        if unlabeled_dataset:
            api_calls = range(min(max_prompt_calls, num_samples_to_generate, len(unlabeled_dataset)))
        else:
            api_calls = range(min(max_prompt_calls, num_samples_to_generate))

        for prompt_call_idx, unlabeled_example_idx in tqdm(
            enumerate(api_calls, start=1), desc="Generating dataset", total=len(api_calls)
        ):
            fewshot_examples = None
            unlabeled_example = None
            invocation_context = None
            prompt_labels = None

            if prompt_template.label_options:
                prompt_labels = choice(prompt_template.label_options, 1)[0]

            if fewshot_dataset:
                prompt_labels, fewshot_examples = self._sample_fewshot_examples(
                    prompt_template, fewshot_dataset, fewshot_examples_per_class, fewshot_label_sampling_strategy,
                    fewshot_sampling_column
                )

            prompt_text = prompt_template.get_prompt_text(prompt_labels, fewshot_examples)

            if unlabeled_dataset:
                unlabeled_example = unlabeled_dataset[unlabeled_example_idx]
                invocation_context = prompt_template.filter_example_by_columns(
                    unlabeled_example, prompt_template.fewshot_example_columns
                )

            prediction = self._try_generate(prompt_text, invocation_context)

            if prediction is None:
                current_tries_left -= 1
                logger.warning(f"Could not generate example for prompt {prompt_text}.")
                if current_tries_left == 0:
                    logger.warning(
                        f"Max tries ({self._max_tries}) exceeded. Returning generated dataset with"
                        f" {len(generated_dataset)} examples."
                    )
                    break

            if len(prediction) == 1:
                prediction = prediction[0]

            # If we have a target variable, we re-use the relevant columns of the input example
            # and add the prediction to the generated dataset
            if prompt_template.generate_data_for_column and unlabeled_example:
                generated_sample = prompt_template.filter_example_by_columns(
                    unlabeled_example, prompt_template.fewshot_example_columns
                )

                for key, value in generated_sample.items():
                    generated_dataset[key].append(value)

                # Try to safely convert the prediction to the type of the target variable
                if not prompt_template.generate_data_for_column[0] in unlabeled_example:
                    prediction = self._convert_prediction(
                        prediction, type(prompt_template.generate_data_for_column[0])
                    )

                generated_dataset[prompt_template.generate_data_for_column[0]].append(prediction)

            else:
                generated_dataset[prompt_template.DEFAULT_COLUMN[0]].append(prediction)

            log_entry = {
                "prompt": prompt_text,
                "invocation_context": invocation_context,
                "prediction": prediction,
                "target": prompt_template.generate_data_for_column[0]
                if prompt_template.generate_data_for_column
                else prompt_template.DEFAULT_COLUMN[0],
            }
            with open(current_log_file, "a", encoding="utf-8") as log_file:
                log_file.write(f"{json.dumps(log_entry)}\n")

            if return_unlabeled_dataset:
                for key, value in unlabeled_example.items():
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

        if return_unlabeled_dataset:
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

    @staticmethod
    def _sample_fewshot_examples(
            prompt_template: BasePrompt,
            fewshot_dataset: Dataset,
            fewshot_examples_per_class: int,
            fewshot_label_sampling_strategy: str,
            fewshot_sampling_column: str
    ) -> Tuple[Union[List[str], str], Dataset]:
        if fewshot_label_sampling_strategy == "uniform":
            prompt_labels = choice(prompt_template.label_options, 1)[0]
            fewshot_examples = fewshot_dataset.filter(
                lambda example: example[fewshot_sampling_column] == prompt_labels
            ).shuffle().select(range(fewshot_examples_per_class))

        elif fewshot_label_sampling_strategy == "stratified":
            prompt_labels = prompt_template.label_options
            fewshot_examples = single_label_stratified_sample(
                fewshot_dataset,
                fewshot_sampling_column,
                fewshot_examples_per_class
            )

        else:
            prompt_labels = prompt_template.label_options if prompt_template.label_options else None
            fewshot_examples = fewshot_dataset.shuffle().select(range(fewshot_examples_per_class))

        assert len(fewshot_examples) > 0, f"Could not find any fewshot examples for label(s) {prompt_labels}." \
                                          f"Ensure that labels of fewshot examples match the label_options " \
                                          f"from the prompt."

        return prompt_labels, fewshot_examples

    @staticmethod
    def _assert_fewshot_dataset_matches_prompt(prompt_template: BasePrompt, fewshot_dataset: Dataset) -> None:
        """Asserts that the prompt template is valid and all columns are present in the fewshot dataset."""
        assert all(
            field in fewshot_dataset.column_names for field in prompt_template.relevant_columns_for_fewshot_examples
        ), "Not all required variables of the prompt template occur in the support examples."
