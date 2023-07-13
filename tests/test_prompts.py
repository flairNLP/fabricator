import unittest

from datasets import load_dataset, QuestionAnsweringExtractive, TextClassification, Summarization

from ai_dataset_generator.prompts import (
    TextLabelPrompt,
    ClassLabelPrompt,
    TokenLabelPrompt,
    infer_prompt_from_task_template,
)


class TestTextLabelPrompt(unittest.TestCase):
    """Testcase for TextLabelPrompt"""

    def setUp(self) -> None:
        self.dataset = (
            load_dataset("opus100", language_pair="de-nl", split="test")
            .flatten()
            .rename_columns({"translation.de": "german", "translation.nl": "dutch"})
        )

    def test_initialization_single_columns(self):
        """Test initialization"""
        text_label_prompt = TextLabelPrompt("test_input", "test_target")
        self.assertEqual(text_label_prompt.variables_for_examples, ["test_input", "test_target"])
        self.assertEqual(type(text_label_prompt.input_variables), list)
        self.assertEqual(len(text_label_prompt.input_variables), 1)

    def test_initialization_multiple_columns(self):
        """Test initialization"""
        text_label_prompt = TextLabelPrompt(["test_input1", "test_input2"], "test_target")
        self.assertEqual(text_label_prompt.variables_for_examples, ["test_input1", "test_input2", "test_target"])
        self.assertEqual(type(text_label_prompt.input_variables), list)
        self.assertEqual(len(text_label_prompt.input_variables), 2)

    def test_target_formatting_template(self):
        """Test initialization"""
        class_label_prompt = TextLabelPrompt("test_input", "test_target")
        self.assertEqual(class_label_prompt.target_formatting_template, "Test_input: {test_input}\nTest_target: ")

    def test_formatting(self):
        input_variable = "german"
        target_variable = "dutch"

        fewshot_examples = self.dataset.select([1, 2, 3])

        prompt = TextLabelPrompt(
            input_variables=input_variable,
            target_variable=target_variable,
            task_description="Given a german phrase, translate it into dutch.",
        )

        raw_prompt = prompt.get_prompt_text(fewshot_examples)
        self.assertEqual(
            raw_prompt,
            "Given a german phrase, translate it into dutch.\n\nGerman: Er mischt sich normalerweise nicht so früh ein.\nDutch: Hij begint normaal pas na z'n partijtje golf.\n\nGerman: Ergebnisse zur ersten Übertragung von Vermögenswerten gemäß dem Beschluss vom November 2010\nDutch: Resultaten met betrekking tot de eerste overdracht van activa overeenkomstig het besluit van november 2010\n\nGerman: gestützt auf die Verordnung (EG) Nr. 1493/1999 des Rates vom 17. Mai 1999 über die gemeinsame Marktorganisation für Wein(1), zuletzt geändert durch die Verordnung (EG) Nr. 2585/2001(2), insbesondere auf die Artikel 10, 15 und 80,\nDutch: Gelet op Verordening (EG) nr. 1493/1999 van de Raad van 17 mei 1999 houdende een gemeenschappelijke ordening van de wijnmarkt(1), laatstelijk gewijzigd bij Verordening (EG) nr. 2585/2001(2), en met name op de artikelen 10, 15 en 80,\n\nGerman: {german}\nDutch: ",
        )


class TestClassLabelPrompt(unittest.TestCase):
    """Testcase for ClassLabelPrompt"""

    def setUp(self) -> None:
        self.dataset = load_dataset("trec", split="train")

    def test_mandatory_label_options(self):
        """Test initialization without mandatory label options"""
        with self.assertRaises(TypeError):
            ClassLabelPrompt("test_input", "test_target")

    def test_warning_label_options_and_not_formatable_description(self):
        """Test initialization with mandatory label options but no formattable description"""
        with self.assertLogs(level="WARNING") as log:
            ClassLabelPrompt(
                "test_input",
                "test_target",
                label_options=["test_label1", "test_label2"],
                task_description="This is not formattable.",
            )
            self.assertEqual(len(log.output), 1)
            self.assertEqual(len(log.records), 1)
            self.assertIn("{label_options} is not found in the task_description.", log.output[0])

    def test_initialization_single_columns(self):
        """Test initialization with single column"""
        class_label_prompt = ClassLabelPrompt(
            "test_input", "test_target", label_options=["test_label1", "test_label2"]
        )
        self.assertEqual(class_label_prompt.variables_for_examples, ["test_input", "test_target"])
        self.assertEqual(type(class_label_prompt.input_variables), list)
        self.assertEqual(len(class_label_prompt.input_variables), 1)

    def test_initialization_multiple_columns(self):
        """Test initialization with multiple columns"""
        class_label_prompt = ClassLabelPrompt(
            ["test_input1", "test_input2"], "test_target", label_options=["test_label1", "test_label2"]
        )
        self.assertEqual(class_label_prompt.variables_for_examples, ["test_input1", "test_input2", "test_target"])
        self.assertEqual(type(class_label_prompt.input_variables), list)
        self.assertEqual(len(class_label_prompt.input_variables), 2)

    def test_target_formatting_template(self):
        """Test formatting template"""
        class_label_prompt = ClassLabelPrompt(
            "test_input", "test_target", label_options=["test_label1", "test_label2"]
        )
        self.assertEqual(class_label_prompt.target_formatting_template, "Test_input: {test_input}\nTest_target: ")

    def test_formatting(self):
        id2label = dict(enumerate(self.dataset.features["coarse_label"].names))
        fewshot_examples = self.dataset.select([1, 2, 3])

        prompt = ClassLabelPrompt(
            input_variables="text",
            target_variable="coarse_label",
            label_options=id2label,
        )
        self.assertIn(", ".join([f"{k}: {v}" for k, v in id2label.items()]), prompt.task_description)

        raw_prompt = prompt.get_prompt_text(fewshot_examples)
        self.assertEqual(
            raw_prompt,
            "Given the following classification examples, annotate the unlabeled example with a prediction that must correspond to exactly one of the following labels: 0: ABBR, 1: ENTY, 2: DESC, 3: HUM, 4: LOC, 5: NUM.\n\nText: What films featured the character Popeye Doyle ?\nCoarse_label: 1\n\nText: How can I find a list of celebrities ' real names ?\nCoarse_label: 2\n\nText: What fowl grabs the spotlight after the Chinese Year of the Monkey ?\nCoarse_label: 1\n\nText: {text}\nCoarse_label: ",
        )


class TestTokenLabelPrompt(unittest.TestCase):
    """Testcase for TokenLabelPrompt"""

    def setUp(self) -> None:
        self.dataset = load_dataset("conll2003", split="train")

    def test_mandatory_label_options(self):
        """Test initialization without mandatory label options"""
        with self.assertRaises(TypeError):
            TokenLabelPrompt("test_input", "test_target")

    def test_warning_label_options_and_not_formatable_description(self):
        """Test initialization with mandatory label options but no formattable description"""
        with self.assertLogs(level="WARNING") as log:
            TokenLabelPrompt(
                "test_input",
                "test_target",
                label_options=["test_label1", "test_label2"],
                task_description="This is not formattable.",
            )
            self.assertEqual(len(log.output), 1)
            self.assertEqual(len(log.records), 1)
            self.assertIn("{label_options} is not found in the task_description.", log.output[0])

    def test_initialization_single_columns(self):
        """Test initialization with single column"""
        token_label_prompt = TokenLabelPrompt(
            "test_input", "test_target", label_options=["test_label1", "test_label2"]
        )
        self.assertEqual(token_label_prompt.variables_for_examples, ["test_input", "test_target"])
        self.assertEqual(type(token_label_prompt.input_variables), list)
        self.assertEqual(len(token_label_prompt.input_variables), 1)

    def test_initialization_multiple_columns(self):
        """Test initialization with multiple columns"""
        token_label_prompt = TokenLabelPrompt(
            ["test_input1", "test_input2"], "test_target", label_options=["test_label1", "test_label2"]
        )
        self.assertEqual(token_label_prompt.variables_for_examples, ["test_input1", "test_input2", "test_target"])
        self.assertEqual(type(token_label_prompt.input_variables), list)
        self.assertEqual(len(token_label_prompt.input_variables), 2)

    def test_target_formatting_template(self):
        """Test formatting template"""
        token_label_prompt = TokenLabelPrompt(
            "test_input", "test_target", label_options=["test_label1", "test_label2"]
        )
        self.assertEqual(token_label_prompt.target_formatting_template, "Test_input: {test_input}\nTest_target: ")

    def test_formatting(self):
        id2label = dict(enumerate(self.dataset.features["ner_tags"].feature.names))
        fewshot_examples = self.dataset.select([1, 2, 3])

        prompt = ClassLabelPrompt(
            input_variables="tokens",
            target_variable="ner_tags",
            label_options=id2label,
        )
        self.assertIn(", ".join([f"{k}: {v}" for k, v in id2label.items()]), prompt.task_description)

        raw_prompt = prompt.get_prompt_text(fewshot_examples)
        self.assertEqual(
            raw_prompt,
            "Given the following classification examples, annotate the unlabeled example with a prediction that must correspond to exactly one of the following labels: 0: O, 1: B-PER, 2: I-PER, 3: B-ORG, 4: I-ORG, 5: B-LOC, 6: I-LOC, 7: B-MISC, 8: I-MISC.\n\nTokens: ['Peter', 'Blackburn']\nNer_tags: [1, 2]\n\nTokens: ['BRUSSELS', '1996-08-22']\nNer_tags: [5, 0]\n\nTokens: ['The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German', 'advice', 'to', 'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad', 'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.']\nNer_tags: [0, 3, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n\nTokens: {tokens}\nNer_tags: ",
        )

    def test_auto_infer_text_label_prompt(self):
        """Test auto inference of TextLabelPrompt from QuestionAnsweringExtractive task template"""
        task_template = QuestionAnsweringExtractive()
        prompt = infer_prompt_from_task_template(task_template)
        self.assertIsInstance(prompt, TextLabelPrompt)
        self.assertEqual(prompt.input_variables, ["context", "question"])
        self.assertEqual(prompt.target_variable, "answer")

    def test_auto_infer_class_label_prompt(self):
        """Test auto inference of ClassLabelPrompt from TextClassification task template"""
        task_template = TextClassification()
        task_template.label_schema["labels"].names = ["neg", "pos"]
        prompt = infer_prompt_from_task_template(task_template)
        self.assertIsInstance(prompt, ClassLabelPrompt)
        self.assertEqual(prompt.input_variables, ["text"])
        self.assertEqual(prompt.target_variable, "labels")

    def test_auto_infer_fails_for_unsupported_task(self):
        """Test auto inference of prompt fails for unsupported task template Summarization"""
        with self.assertRaises(ValueError):
            infer_prompt_from_task_template(Summarization())
