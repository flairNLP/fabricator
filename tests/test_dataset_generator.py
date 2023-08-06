import unittest

from datasets import Dataset, load_dataset

from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt
from fabricator.dataset_transformations.text_classification import convert_label_ids_to_texts


class TestDatasetGenerator(unittest.TestCase):
    """Testcase for Prompts"""

    def setUp(self) -> None:
        """Set up test dataset"""
        self.text_classification_dataset = Dataset.from_dict({
            "text": ["This movie is great!", "This movie is bad!"],
            "label": ["positive", "negative"]
        })

        # We are using dummy respones here, because we are not testing the LLM itself.
        self.generator = DatasetGenerator(None)

    def test_simple_generation(self):
        """Test simple generation without fewshot examples."""
        prompt = BasePrompt(
            task_description="Generate a short movie review.",
        )

        generated_dataset = self.generator.generate(
            prompt_template=prompt,
            max_prompt_calls=2,
            dummy_response="A dummy movie review."
        )

        self.assertEqual(len(generated_dataset), 2)
        self.assertEqual(generated_dataset.features["text"].dtype, "string")
        self.assertIn("text", generated_dataset.features)

    def test_simple_generation_with_label_options(self):
        """Test simple generation without fewshot examples with label options."""
        prompt = BasePrompt(
            task_description="Generate a short {} movie review.",
            label_options=["positive", "negative"],
        )

        generated_dataset = self.generator.generate(
            prompt_template=prompt,
            max_prompt_calls=2,
            dummy_response="A dummy movie review."
        )

        self.assertEqual(len(generated_dataset), 2)
        self.assertEqual(generated_dataset.features["text"].dtype, "string")
        self.assertIn("text", generated_dataset.features)

    def test_generation_with_fewshot_examples(self):
        label_options = ["positive", "negative"]

        prompt = BasePrompt(
            task_description="Generate a {} movie review.",
            label_options=label_options,
            generate_data_for_column="text",
        )

        generated_dataset = self.generator.generate(
            prompt_template=prompt,
            fewshot_dataset=self.text_classification_dataset,
            fewshot_examples_per_class=1,
            fewshot_sampling_strategy="uniform",
            fewshot_sampling_column="label",
            max_prompt_calls=2,
            dummy_response="A dummy movie review."
        )

        self.assertEqual(len(generated_dataset), 2)
        self.assertEqual(generated_dataset.features["text"].dtype, "string")
        self.assertIn("text", generated_dataset.features)

    def test_annotation_with_fewshot_and_unlabeled_examples(self):
        label_options = ["positive", "negative"]

        unlabeled_dataset = Dataset.from_dict({
            "text": ["This movie was a blast!", "This movie was not bad!"],
        })

        prompt = BasePrompt(
            task_description="Annotate movie reviews as either: {}.",
            label_options=label_options,
            generate_data_for_column="label",
            fewshot_example_columns="text",
        )

        generated_dataset = self.generator.generate(
            prompt_template=prompt,
            fewshot_dataset=self.text_classification_dataset,
            fewshot_examples_per_class=1,
            fewshot_sampling_strategy="stratified",
            unlabeled_dataset=unlabeled_dataset,
            max_prompt_calls=2,
            dummy_response="A dummy movie review."
        )

        self.assertEqual(len(generated_dataset), 2)
        self.assertEqual(generated_dataset.features["text"].dtype, "string")
        self.assertEqual(generated_dataset.features["label"].dtype, "string")
        self.assertIn("text", generated_dataset.features)
        self.assertIn("label", generated_dataset.features)
        self.assertEqual(generated_dataset[0]["text"], "This movie was a blast!")
        self.assertEqual(generated_dataset[1]["text"], "This movie was not bad!")

    def test_tranlation(self):
        fewshot_dataset = Dataset.from_dict({
            "german": ["Der Film ist großartig!", "Der Film ist schlecht!"],
            "english": ["This movie is great!", "This movie is bad!"],
        })

        unlabeled_dataset = Dataset.from_dict({
            "english": ["This movie was a blast!", "This movie was not bad!"],
        })

        prompt = BasePrompt(
            task_description="Translate to german:",  # Since we do not have a label column,
            # we can just specify the task description
            generate_data_for_column="german",
            fewshot_example_columns="english",
        )

        generated_dataset = self.generator.generate(
            prompt_template=prompt,
            fewshot_dataset=fewshot_dataset,
            fewshot_examples_per_class=2,  # Take both fewshot examples per prompt
            fewshot_sampling_strategy=None,
            # Since we do not have a class label column, we can just set this to None
            # (default)
            unlabeled_dataset=unlabeled_dataset,
            max_prompt_calls=2,
            dummy_response="A dummy movie review."
        )

        self.assertEqual(len(generated_dataset), 2)
        self.assertEqual(generated_dataset.features["english"].dtype, "string")
        self.assertEqual(generated_dataset.features["german"].dtype, "string")
        self.assertIn("english", generated_dataset.features)
        self.assertIn("german", generated_dataset.features)

    def test_textual_similarity(self):
        dataset = load_dataset("glue", "mrpc", split="train")
        dataset, label_options = convert_label_ids_to_texts(dataset, "label", return_label_options=True)  # convert the
        # label ids to text labels and return the label options

        fewshot_dataset = dataset.select(range(10))
        unlabeled_dataset = dataset.select(range(10, 20))

        prompt = BasePrompt(
            task_description="Annotate the sentence pair whether it is: {}",
            label_options=label_options,
            generate_data_for_column="label",
            fewshot_example_columns=["sentence1", "sentence2"],
        )

        generated_dataset, original_dataset = self.generator.generate(
            prompt_template=prompt,
            fewshot_dataset=fewshot_dataset,
            fewshot_examples_per_class=1,
            fewshot_sampling_column="label",
            fewshot_sampling_strategy="stratified",
            unlabeled_dataset=unlabeled_dataset,
            max_prompt_calls=2,
            return_unlabeled_dataset=True,
            dummy_response="A dummy movie review."
        )

        self.assertEqual(len(generated_dataset), 2)
        self.assertEqual(generated_dataset.features["sentence1"].dtype, "string")
        self.assertEqual(generated_dataset.features["sentence2"].dtype, "string")
        self.assertEqual(generated_dataset.features["label"].dtype, "string")

    def test_sampling_all_fewshot_examples(self):
        """Test sampling all fewshot examples"""
        prompt = BasePrompt(
            task_description="Generate a short movie review.",
        )

        prompt_labels, fewshot_examples = self.generator._sample_fewshot_examples(
            prompt_template=prompt,
            fewshot_dataset=self.text_classification_dataset,
            fewshot_sampling_strategy=None,
            fewshot_examples_per_class=None,
            fewshot_sampling_column=None,
        )

        self.assertEqual(prompt_labels, None)
        self.assertEqual(len(fewshot_examples), 2)

    def test_sampling_all_fewshot_examples_with_label_options(self):
        """Test sampling all fewshot examples with label options"""
        prompt = BasePrompt(
            task_description="Generate a short movie review: {}.",
            label_options=["positive", "negative"],
        )

        prompt_labels, fewshot_examples = self.generator._sample_fewshot_examples(
            prompt_template=prompt,
            fewshot_dataset=self.text_classification_dataset,
            fewshot_sampling_strategy=None,
            fewshot_examples_per_class=None,
            fewshot_sampling_column=None,
        )

        prompt_text = prompt.task_description.format(prompt_labels)
        self.assertIn("positive", prompt_text)
        self.assertIn("negative", prompt_text)

    def test_sampling_uniform_fewshot_examples(self):
        """Test uniform sampling fewshot examples"""
        prompt = BasePrompt(
            task_description="Generate a short movie review: {}.",
            label_options=["positive", "negative"],
        )

        prompt_labels, fewshot_examples = self.generator._sample_fewshot_examples(
            prompt_template=prompt,
            fewshot_dataset=self.text_classification_dataset,
            fewshot_sampling_strategy="uniform",
            fewshot_examples_per_class=1,
            fewshot_sampling_column="label",
        )

        self.assertEqual(len(fewshot_examples), 1)
        self.assertIn(prompt_labels, ["positive", "negative"])

    def test_sampling_stratified_fewshot_examples(self):
        """Test stratified sampling fewshot examples"""
        larger_fewshot_dataset = Dataset.from_dict({
            "text": ["This movie is great!", "This movie is bad!", "This movie is great!", "This movie is bad!"],
            "label": ["positive", "negative", "positive", "negative"]
        })

        prompt = BasePrompt(
            task_description="Generate a short movie review: {}.",
            label_options=["positive", "negative"],
        )

        prompt_labels, fewshot_examples = self.generator._sample_fewshot_examples(
            prompt_template=prompt,
            fewshot_dataset=larger_fewshot_dataset,
            fewshot_sampling_strategy="stratified",
            fewshot_examples_per_class=1,
            fewshot_sampling_column="label",
        )

        self.assertEqual(len(fewshot_examples), 2)
        self.assertEqual(len(set(fewshot_examples["label"])), 2)
        self.assertEqual(len(prompt_labels), 2)

    def test_sampling_uniform_fewshot_examples_without_number_of_examples(self):
        """Test failure of uniform sampling fewshot examples if attributes are missing"""
        prompt = BasePrompt(
            task_description="Generate a short movie review: {}.",
            label_options=["positive", "negative"],
        )

        with self.assertRaises(KeyError):
            prompt_labels, fewshot_examples = self.generator._sample_fewshot_examples(
                prompt_template=prompt,
                fewshot_dataset=self.text_classification_dataset,
                fewshot_sampling_strategy="uniform",
                fewshot_examples_per_class=None,
                fewshot_sampling_column=None,
            )

    def test_sampling_uniform_fewshot_examples_with_number_of_examples_without_sampling_column(self):
        """Test failure of uniform sampling fewshot examples if attributes are missing"""
        prompt = BasePrompt(
            task_description="Generate a short movie review: {}.",
            label_options=["positive", "negative"],
        )

        with self.assertRaises(KeyError):
            prompt_labels, fewshot_examples = self.generator._sample_fewshot_examples(
                prompt_template=prompt,
                fewshot_dataset=self.text_classification_dataset,
                fewshot_sampling_strategy="uniform",
                fewshot_examples_per_class=1,
                fewshot_sampling_column=None,
            )

    def test_dummy_response(self):

        prompt = BasePrompt(
            task_description="Generate a short movie review.",
        )
        generated_dataset = self.generator.generate(
            prompt_template=prompt,
            max_prompt_calls=2,
            dummy_response=lambda _: "This is a dummy movie review."
        )

        self.assertEqual(len(generated_dataset), 2)
        self.assertEqual(generated_dataset.features["text"].dtype, "string")
        self.assertIn("text", generated_dataset.features)
        self.assertEqual(generated_dataset[0]["text"], "This is a dummy movie review.")
        self.assertEqual(generated_dataset[1]["text"], "This is a dummy movie review.")


        generated_dataset = self.generator.generate(
            prompt_template=prompt,
            max_prompt_calls=2,
            dummy_response="This is a dummy movie review as a string."
        )

        self.assertEqual(len(generated_dataset), 2)
        self.assertEqual(generated_dataset.features["text"].dtype, "string")
        self.assertIn("text", generated_dataset.features)
        self.assertEqual(generated_dataset[0]["text"], "This is a dummy movie review as a string.")
        self.assertEqual(generated_dataset[1]["text"], "This is a dummy movie review as a string.")
