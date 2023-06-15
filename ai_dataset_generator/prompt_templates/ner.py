"""
Prompting Templates for Sequence Labeling.

The tagging scheme follows CoNNL-2003 (https://aclanthology.org/W03-0419/) standard and
the commonly used Huggingface tag representation.

"""

from typing import Dict, Optional

from ai_dataset_generator.prompt_templates import AnnotationPrompt


class SequenceLabelingPrompt(AnnotationPrompt):
    tags: Optional[dict] = None

    def __init__(self):
        super().__init__(
            task_description=f"Given a list of tokens, generate a list of annotations for every token with according to their label based on following tags: {self.formatted_tags()}",
            support_set_variables=["tokens", "annotations"],
            support_set_formatting_template="""Text: {tokens}\nAnnotations: {annotations}""",
            annotation_variables=["tokens"],
            annotation_formatting_template="""Text: {tokens}\nAnnotations: """,
        )

    def formatted_tags(self):
        raise NotImplementedError("Sequence labeling task should provide a dictionary `tags` in the form {Label: ID}")

    def format_variables(self, variable: str, value: str) -> Dict[str, str]:
        return {variable: value}


class NamedEntityAnnotationPrompt(AnnotationPrompt):
    tags = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}

    def __init__(self):
        try:
            from nltk.tokenize import word_tokenize

            word_tokenize("")
        except ImportError:
            print("Annotation for Named Entity Recognition requires 'nltk'. Please install with 'pip install nltk'")
            exit()
        except LookupError as e:
            raise e

        super().__init__(
            task_description=f"Given a list of tokens, generate a list of annotations for every token with according to their entity based on following NER tags: {self.formatted_tags()}",
            support_set_variables=["tokens", "annotations"],
            support_set_formatting_template="""Text: {tokens}\nAnnotations: {annotations}""",
            annotation_variables=["tokens"],
            annotation_formatting_template="""Text: {tokens}\nAnnotations: """,
        )

    def formatted_tags(self):
        return ", ".join([f"{k}: {v}" for k, v in self.tags.items()])


class PartOfSpeechAnnotationPrompt(AnnotationPrompt):
    tags = {
        '"': 0,
        "''": 1,
        "#": 2,
        "$": 3,
        "(": 4,
        ")": 5,
        ",": 6,
        ".": 7,
        ":": 8,
        "``": 9,
        "CC": 10,
        "CD": 11,
        "DT": 12,
        "EX": 13,
        "FW": 14,
        "IN": 15,
        "JJ": 16,
        "JJR": 17,
        "JJS": 18,
        "LS": 19,
        "MD": 20,
        "NN": 21,
        "NNP": 22,
        "NNPS": 23,
        "NNS": 24,
        "NN|SYM": 25,
        "PDT": 26,
        "POS": 27,
        "PRP": 28,
        "PRP$": 29,
        "RB": 30,
        "RBR": 31,
        "RBS": 32,
        "RP": 33,
        "SYM": 34,
        "TO": 35,
        "UH": 36,
        "VB": 37,
        "VBD": 38,
        "VBG": 39,
        "VBN": 40,
        "VBP": 41,
        "VBZ": 42,
        "WDT": 43,
        "WP": 44,
        "WP$": 45,
        "WRB": 46,
    }

    def __init__(self):
        super().__init__(
            task_description=f"Given a list of tokens, generate a list of annotations for every token with according to their entity based on following Part-of-Speech tags: {self.formatted_tags()}",
            support_set_variables=["tokens", "annotations"],
            support_set_formatting_template="""Text: {tokens}\nAnnotations: {annotations}""",
            annotation_variables=["tokens"],
            annotation_formatting_template="""Text: {tokens}\nAnnotations: """,
        )

    def formatted_tags(self):
        return ", ".join([f"{k}: {v}" for k, v in self.tags.items()])


class ChunkingAnnotationPrompt(AnnotationPrompt):
    tags = {
        "O": 0,
        "B-ADJP": 1,
        "I-ADJP": 2,
        "B-ADVP": 3,
        "I-ADVP": 4,
        "B-CONJP": 5,
        "I-CONJP": 6,
        "B-INTJ": 7,
        "I-INTJ": 8,
        "B-LST": 9,
        "I-LST": 10,
        "B-NP": 11,
        "I-NP": 12,
        "B-PP": 13,
        "I-PP": 14,
        "B-PRT": 15,
        "I-PRT": 16,
        "B-SBAR": 17,
        "I-SBAR": 18,
        "B-UCP": 19,
        "I-UCP": 20,
        "B-VP": 21,
        "I-VP": 22,
    }

    def __init__(self):
        super().__init__(
            task_description=f"Given a list of tokens, generate a list of annotations for every token with according to their entity based on following Chunking tags: {self.formatted_tags()}",
            support_set_variables=["tokens", "annotations"],
            support_set_formatting_template="""Text: {tokens}\nAnnotations: {annotations}""",
            annotation_variables=["tokens"],
            annotation_formatting_template="""Text: {tokens}\nAnnotations: """,
        )

    def formatted_tags(self):
        return ", ".join([f"{k}: {v}" for k, v in self.tags.items()])
