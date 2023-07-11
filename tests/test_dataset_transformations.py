import unittest

from datasets import load_dataset

from ai_dataset_generator.prompts import ClassLabelPrompt, TokenLabelPrompt
from ai_dataset_generator.dataset_transformations.text_classification import *
from ai_dataset_generator.dataset_transformations.token_classification import *
from ai_dataset_generator.dataset_transformations.question_answering import *


class TestTransformationsTextClassification(unittest.TestCase):
    """Testcase for ClassLabelPrompt"""
    def setUp(self) -> None:
        self.input_variable = "text"
        self.target_variable = "coarse_label"
        self.dataset = load_dataset("trec", split="train")

    def test_label_ids_to_textual_label(self):
        """Test transformation output only"""
        dataset, label_options = convert_label_ids_to_text(self.dataset, self.target_variable)
        self.assertEqual(len(label_options), 6)
        self.assertEqual(set(label_options), set(self.dataset.features[self.target_variable].names))
        self.assertEqual(type(dataset[0][self.target_variable]), str)
        self.assertNotEqual(type(dataset[0][self.target_variable]), int)
        self.assertIn(dataset[0][self.target_variable], label_options)

    def test_formatting_with_textual_labels(self):
        """Test formatting with textual labels"""
        dataset, label_options = convert_label_ids_to_text(self.dataset, self.target_variable)
        fewshot_examples = dataset.select([1, 2, 3])
        prompt = ClassLabelPrompt(
            input_variables=self.input_variable,
            target_variable=self.target_variable,
            label_options=label_options,
        )
        raw_prompt = prompt.get_prompt_text(fewshot_examples)
        self.assertIn("Text: What films featured the character Popeye Doyle ?\nCoarse_label: ENTY", raw_prompt)
        for label in label_options:
            self.assertIn(label, prompt.task_description)

    def test_expanded_textual_labels(self):
        """Test formatting with expanded textual labels"""
        extended_mapping = {
            "DESC": "Description",
            "ENTY": "Entity",
            "ABBR": "Abbreviation",
            "HUM": "Human",
            "NUM": "Number",
            "LOC": "Location"
        }
        dataset, label_options = convert_label_ids_to_text(
            self.dataset,
            self.target_variable,
            expanded_label_mapping=extended_mapping
        )
        self.assertIn("Location", label_options)
        self.assertNotIn("LOC", label_options)
        fewshot_examples = dataset.select([1, 2, 3])
        prompt = ClassLabelPrompt(
            input_variables=self.input_variable,
            target_variable=self.target_variable,
            label_options=label_options,
        )
        raw_prompt = prompt.get_prompt_text(fewshot_examples)
        self.assertIn("Text: What films featured the character Popeye Doyle ?\nCoarse_label: Entity", raw_prompt)
        self.assertNotIn("ENTY", raw_prompt)
        for label in label_options:
            self.assertIn(label, prompt.task_description)

    def test_textual_labels_to_label_ids(self):
        """Test conversion back to label ids"""
        dataset, label_options = convert_label_ids_to_text(self.dataset, self.target_variable)
        self.assertIn(dataset[0][self.target_variable], label_options)
        dataset = dataset.class_encode_column(self.target_variable)
        self.assertIn(dataset[0][self.target_variable], range(len(label_options)))

    def test_false_inputs_raises_error(self):
        """Test that false inputs raise errors"""
        with self.assertRaises(AttributeError):
            dataset, label_options = convert_label_ids_to_text(self.target_variable, self.dataset)


class TestTransformationsTokenClassification(unittest.TestCase):
    """Testcase for TokenLabelTransformations"""
    def setUp(self) -> None:
        self.input_variable = "tokens"
        self.target_variable = "ner_tags"
        self.dataset = load_dataset("conll2003", split="train")
        self.id2label = dict(enumerate(self.dataset.features[self.target_variable].feature.names))

    def test_bio_tokens_to_spans(self):
        """Test transformation output only (BIO to spans)"""
        dataset, label_options = token_labels_to_spans(self.dataset, self.input_variable, self.target_variable, self.id2label)
        self.assertEqual(len(label_options), 4)
        self.assertEqual(type(dataset[0][self.target_variable]), str)
        self.assertNotEqual(type(dataset[0][self.target_variable]), int)
        labels = [spans.split(LABEL2ENTITY_SEPARATOR, 1)[0].strip() for spans in dataset[0][self.target_variable].split(LABEL_SEPARATOR)]
        for label in labels:
            self.assertIn(label, label_options)

    def test_formatting_with_span_labels(self):
        """Test formatting with span labels"""
        dataset, label_options = token_labels_to_spans(self.dataset, self.input_variable, self.target_variable, self.id2label)
        fewshot_examples = dataset.select([1, 2, 3])
        prompt = TokenLabelPrompt(
            input_variables=self.input_variable,
            target_variable=self.target_variable,
            label_options=label_options,
        )
        raw_prompt = prompt.get_prompt_text(fewshot_examples)
        self.assertIn("PER -> Peter Blackburn", raw_prompt)
        self.assertIn("LOC -> BRUSSELS", raw_prompt)
        for label in label_options:
            self.assertIn(label, prompt.task_description)

    def test_expanded_textual_labels(self):
        """Test formatting with expanded textual labels"""
        extended_mapping = {
            "PER": "person",
            "LOC": "location",
            "ORG": "organization",
            "MISC": "misceallaneous"
        }
        id2label = replace_token_labels(self.id2label, extended_mapping)
        self.assertIn("B-location", id2label.values())
        self.assertIn("I-person", id2label.values())
        self.assertNotIn("B-LOC", id2label.values())
        self.assertNotIn("I-MISC", id2label.values())
        dataset, label_options = token_labels_to_spans(self.dataset, self.input_variable, self.target_variable, id2label)
        fewshot_examples = dataset.select([1, 2, 3])
        prompt = TokenLabelPrompt(
            input_variables=self.input_variable,
            target_variable=self.target_variable,
            label_options=label_options,
        )
        raw_prompt = prompt.get_prompt_text(fewshot_examples)
        self.assertIn("person -> Peter Blackburn", raw_prompt)
        self.assertNotIn("PER", raw_prompt)
        for label in label_options:
            self.assertIn(label, prompt.task_description)

    def test_textual_labels_to_label_ids(self):
        """Test conversion back to label ids on token-level"""
        dataset, label_options = token_labels_to_spans(self.dataset, self.input_variable, self.target_variable, self.id2label)
        self.assertEqual(dataset[0][self.target_variable], "ORG -> EU\nMISC -> German, British")
        dataset = dataset.select(range(10))
        dataset = spans_to_token_labels(dataset, self.input_variable, self.target_variable, self.id2label)
        for label in dataset[0][self.target_variable]:
            self.assertIn(label, self.id2label.keys())

    def test_false_inputs_raises_error(self):
        """Test that false inputs raise errors"""
        with self.assertRaises(AttributeError):
            dataset, label_options = token_labels_to_spans(self.target_variable, self.dataset, self.input_variable, self.id2label)

        with self.assertRaises(AttributeError):
            dataset, label_options = token_labels_to_spans(self.id2label, self.target_variable, self.dataset, self.input_variable,)


class TestTransformationsQuestionAnswering(unittest.TestCase):
    """Testcase for QA Transformations"""
    def setUp(self) -> None:
        self.input_variable = ["context", "question"]
        self.target_variable = "answer"
        self.dataset = load_dataset("squad_v2", split="train")

    def test_squad_preprocessing(self):
        """Test transformation from squad fromat into flat structure"""
        self.assertEqual(type(self.dataset[0]["answers"]), dict)
        dataset = preprocess_squad_format(self.dataset.select(range(30)))
        self.assertEqual(type(dataset[0]["answer"]), str)
        with self.assertRaises(KeyError):
             x = dataset[0]["answer_start"]


    def test_squad_postprocessing(self):
        """Test transformation flat structure into squad format"""
        dataset = preprocess_squad_format(self.dataset.select(range(50)))
        dataset = postprocess_squad_format(dataset)
        self.assertEqual(type(dataset[0]["answer"]), dict)
        self.assertIn("start", dataset[0]["answer"])
