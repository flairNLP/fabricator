import os
from datasets import load_dataset
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator, BasePrompt
from ai_dataset_generator.dataset_transformations.token_classification import convert_token_labels_to_spans


def run():
    fewshot_dataset = load_dataset("conll2003", split="train")
    fewshot_dataset, label_options = convert_token_labels_to_spans(
        fewshot_dataset,
        "tokens",
        "ner_tags",
        expanded_label_mapping={
            0: "O",
            1: "B-person",
            2: "I-person",
            3: "B-organization",
            4: "I-organization",
            5: "B-location",
            6: "I-location",
            7: "B-miscellaneous",
            8: "I-miscellaneous",
        }
    )

    annotation_dataset = load_dataset("conll2003", split="validation")
    annotation_dataset, label_options = convert_token_labels_to_spans(
        annotation_dataset,
        "tokens",
        "ner_tags",
        expanded_label_mapping={
            0: "O",
            1: "B-person",
            2: "I-person",
            3: "B-organization",
            4: "I-organization",
            5: "B-location",
            6: "I-location",
            7: "B-miscellaneous",
            8: "I-miscellaneous",
        }
    )

    prompt = BasePrompt(
        task_description="Extract the following named entities from the text: {}. "
                         "Your output format must be exactly the same as from the fewshot examples.",
        label_options=label_options,
        generate_data_for_column="ner_tags",
        fewshot_example_columns="tokens",
    )

    prompt_node = PromptNode(
        model_name_or_path="gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_length=500,
    )

    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        prompt_template=prompt,
        fewshot_dataset=fewshot_dataset,
        fewshot_examples_per_class=3,
        unlabeled_dataset=annotation_dataset,
        max_prompt_calls=len(annotation_dataset),
    )

    generated_dataset.push_to_hub("conll-validation-annotated", private=True)


if __name__ == "__main__":
    run()
