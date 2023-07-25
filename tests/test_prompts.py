import unittest

from datasets import load_dataset, Dataset, QuestionAnsweringExtractive, TextClassification, Summarization

from ai_dataset_generator.prompts import (
    BasePrompt,
    infer_prompt_from_task_template,
)


class TestPrompt(unittest.TestCase):
    """Testcase for Prompts"""

    def setUp(self) -> None:
        """Set up test dataset"""
        self.dataset = Dataset.from_dict({
            "text": ["This movie is great!", "This movie is bad!"],
            "label": ["positive", "negative"]
        })

    def test_plain_template(self):
        """Test plain prompt template"""
        prompt_template = BasePrompt(task_description="Generate movies reviews.")
        self.assertEqual(prompt_template.get_prompt_text(), 'Generate movies reviews.\n\ntext: ')
        self.assertEqual(prompt_template.target_formatting_template, "text: ")
        self.assertEqual(prompt_template.generate_data_for_column, None)
        self.assertEqual(prompt_template.fewshot_example_columns, None)

        with self.assertRaises(TypeError):
            prompt_template = BasePrompt()

    def test_template_with_label_options(self):
        """Test prompt template with label options"""
        label_options = ["positive", "negative"]
        prompt_template = BasePrompt(
            task_description="Generate a {} movie review.",
            label_options=label_options,
        )
        self.assertIn("positive", prompt_template.get_prompt_text(label_options[0]))
        self.assertIn("negative", prompt_template.get_prompt_text(label_options[1]))
        self.assertEqual(prompt_template.target_formatting_template, "text: ")

    def test_initialization_only_target_column(self):
        """Test initialization with only target column"""
        prompt_template = BasePrompt(
            task_description="Generate similar movie reviews.",
            generate_data_for_column="text",
        )
        self.assertEqual(prompt_template.relevant_columns_for_fewshot_examples, ["text"])
        self.assertEqual(type(prompt_template.generate_data_for_column), list)
        self.assertEqual(len(prompt_template.generate_data_for_column), 1)
        self.assertEqual(prompt_template.fewshot_example_columns, None)

        prompt_text = 'Generate similar movie reviews.\n\ntext: This movie is great!\n\ntext: ' \
                      'This movie is bad!\n\ntext: '

        self.assertEqual(prompt_template.get_prompt_text(None, self.dataset), prompt_text)

    def test_initialization_target_and_fewshot_columns(self):
        """Test initialization with target and fewshot columns"""
        prompt_template = BasePrompt(
            task_description="Generate movie reviews.",
            generate_data_for_column="label",
            fewshot_example_columns="text"
        )
        self.assertEqual(prompt_template.relevant_columns_for_fewshot_examples, ["text", "label"])
        self.assertEqual(type(prompt_template.generate_data_for_column), list)
        self.assertEqual(len(prompt_template.generate_data_for_column), 1)
        self.assertEqual(type(prompt_template.fewshot_example_columns), list)

        prompt_text = 'Generate movie reviews.\n\ntext: This movie is great!\nlabel: positive\n\n' \
                      'text: This movie is bad!\nlabel: negative\n\ntext: {text}\nlabel: '

        self.assertEqual(prompt_template.get_prompt_text(None, self.dataset), prompt_text)

    def test_initialization_with_multiple_fewshot_columns(self):
        """Test initialization with multiple fewshot columns"""
        text_label_prompt = BasePrompt(
            task_description="Test two fewshot columns.",
            generate_data_for_column="label",
            fewshot_example_columns=["fewshot1", "fewshot2"],
        )
        self.assertEqual(text_label_prompt.relevant_columns_for_fewshot_examples, ["fewshot1", "fewshot2", "label"])
        self.assertEqual(type(text_label_prompt.fewshot_example_columns), list)
        self.assertEqual(len(text_label_prompt.fewshot_example_columns), 2)


class TestDownstreamTasks(unittest.TestCase):
    """Testcase for downstream tasks"""

    def setUp(self) -> None:
        """Set up test datasets"""

        def preprocess_qa(example):
            if example["answer"]:
                example["answer"] = example["answer"].pop()
            else:
                example["answer"] = ""
            return example

        self.text_classification = load_dataset("trec", split="train").select([1, 2, 3])
        self.question_answering = load_dataset("squad", split="train").flatten()\
            .rename_column("answers.text", "answer").map(preprocess_qa).select([1, 2, 3])
        self.ner = load_dataset("conll2003", split="train").select([1, 2, 3])
        self.translation = load_dataset("opus100", language_pair="de-nl", split="test").flatten()\
            .rename_columns({"translation.de": "german", "translation.nl": "dutch"}).select([1, 2, 3])

    def test_translation(self):
        prompt = BasePrompt(
            task_description="Given a german phrase, translate it into dutch.",
            generate_data_for_column="dutch",
            fewshot_example_columns="german",
        )

        raw_prompt = prompt.get_prompt_text(None, self.translation)
        self.assertIn("Marktorganisation für Wein(1)", raw_prompt)
        self.assertIn("Gelet op Verordening (EG) nr. 1493/1999", raw_prompt)
        self.assertIn("na z'n partijtje golf", raw_prompt)
        self.assertIn("dutch: ", raw_prompt)

    def test_text_classification(self):
        label_options = self.text_classification.features["coarse_label"].names

        prompt = BasePrompt(
            task_description="Classify the question into one of the following categories: {}",
            label_options=label_options,
            generate_data_for_column="coarse_label",
            fewshot_example_columns="text",
        )

        raw_prompt = prompt.get_prompt_text(label_options, self.text_classification)
        self.assertIn(", ".join(label_options), raw_prompt)
        self.assertIn("What fowl grabs the spotlight after the Chinese Year of the Monkey ?", raw_prompt)
        self.assertIn("How can I find a list of celebrities ' real names ?", raw_prompt)
        self.assertIn("What films featured the character Popeye Doyle ?", raw_prompt)
        self.assertIn("coarse_label: 2", raw_prompt)

    def test_named_entity_recognition(self):
        label_options = self.ner.features["ner_tags"].feature.names

        prompt = BasePrompt(
            task_description="Classify each token into one of the following categories: {}",
            generate_data_for_column="ner_tags",
            fewshot_example_columns="tokens",
            label_options=label_options,
        )

        raw_prompt = prompt.get_prompt_text(label_options, self.ner)
        self.assertIn(", ".join(label_options), raw_prompt)
        self.assertIn("'BRUSSELS', '1996-08-22'", raw_prompt)
        self.assertIn("'Peter', 'Blackburn'", raw_prompt)
        self.assertIn("3, 4, 0, 0, 0, 0, 0, 0, 7, 0", raw_prompt)
        self.assertIn("ner_tags: [1, 2]", raw_prompt)

    def test_question_answering(self):
        prompt = BasePrompt(
            task_description="Given context and question, answer the question.",
            generate_data_for_column="answer",
            fewshot_example_columns=["context", "question"],
        )

        raw_prompt = prompt.get_prompt_text(None, self.question_answering)
        self.assertIn("answer: the Main Building", raw_prompt)
        self.assertIn("context: Architecturally, the school", raw_prompt)
        self.assertIn("question: The Basilica", raw_prompt)
        self.assertIn("context: {context}", raw_prompt)


class TestAutoInference(unittest.TestCase):
    """Testcase for AutoInference"""

    def test_auto_infer_text_label_prompt(self):
        """Test auto inference of QuestionAnsweringExtractive task template"""
        task_template = QuestionAnsweringExtractive()
        prompt = infer_prompt_from_task_template(task_template)
        self.assertIsInstance(prompt, BasePrompt)
        self.assertEqual(prompt.fewshot_example_columns, ["context", "question"])
        self.assertEqual(prompt.generate_data_for_column, ["answers"])

    def test_auto_infer_class_label_prompt(self):
        """Test auto inference of TextClassification task template"""
        task_template = TextClassification()
        task_template.label_schema["labels"].names = ["neg", "pos"]
        prompt = infer_prompt_from_task_template(task_template)
        self.assertIsInstance(prompt, BasePrompt)
        self.assertEqual(prompt.fewshot_example_columns, ["text"])
        self.assertEqual(prompt.generate_data_for_column, ["labels"])

    def test_auto_infer_fails_for_unsupported_task(self):
        """Test auto inference of prompt fails for unsupported task template Summarization"""
        with self.assertRaises(ValueError):
            infer_prompt_from_task_template(Summarization())
