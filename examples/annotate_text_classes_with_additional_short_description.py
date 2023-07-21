import os
import random
from argparse import ArgumentParser

from datasets import load_dataset
from haystack.nodes import PromptNode

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.dataset_transformations.text_classification import \
    convert_label_ids_to_texts
from ai_dataset_generator.prompts import BasePrompt


def run(arguments):
    """Generate annotations for unlabeled data for a given dataset and split."""
    dataset = load_dataset(arguments.dataset, split=arguments.split)

    expanded_label_mapping = {
        "ABBR": "Abbreviation",
        "DESC": "Description",
        "ENTY": "Entity",
        "HUM": "Human",
        "LOC": "Location",
        "NUM": "Numeric",
    }

    one_sentence_description = {
        "Abbreviation": "A question about abbreviations.",
        "Description": "A question about descriptions.",
        "Entity": "A question about entities",
        "Human": "A question about humans.",
        "Location": "A question about locations.",
        "Numeric": "A question about numbers.",
    }

    dataset, label_options = convert_label_ids_to_texts(
        dataset,
        arguments.label_column,
        expanded_label_mapping=expanded_label_mapping,
        return_label_options=True,
    )
    fewshot_examples = dataset.select(random.sample(range(len(dataset)), arguments.num_fewshot_examples))

    task_description = arguments.task_description.format(
        label_options=", ".join([f"{k}: {v}" for k, v in one_sentence_description.items()])
    )
    prompt = BasePrompt(
        task_description=task_description,
        generate_data_for_column=arguments.label_column,
        fewshot_example_columns=arguments.input_columns,
        label_options=label_options,
    )

    raw_prompt = prompt.get_prompt_text(fewshot_examples)
    print(raw_prompt)

    unlabeled_data = dataset.select(random.sample(range(len(dataset)), 3))
    prompt_node = PromptNode(
        model_name_or_path=arguments.llm,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_length=arguments.max_generation_length,
    )
    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        support_examples=fewshot_examples,  # from above
        unlabeled_examples=unlabeled_data,
        prompt_template=prompt,  # from above
        max_prompt_calls=arguments.max_prompt_calls,  # max number of calls to the LLM
        support_examples_per_prompt=arguments.support_examples_per_prompt,  # number of support examples per prompt
    )

    generated_dataset = generated_dataset.class_encode_column(arguments.label_column)

    if arguments.push_to_hub:
        generated_dataset.push_to_hub("your-first-generated-dataset")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default="text-davinci-003")
    parser.add_argument("--max_generation_length", type=int, default=100)
    parser.add_argument(
        "--task_description",
        type=str,
        default="Classify the question types into one of the following categories: " "{label_options}.",
    )
    parser.add_argument("--dataset", type=str, default="trec")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--input_columns", type=str, nargs="+", default=["text"]
    )  # Column names as they occur in the dataset
    parser.add_argument("--label_column", type=str, default="coarse_label")
    parser.add_argument(
        "--output_format", type=str, default="single_label_classification"
    )  # indicates the output format of the LLM is text
    parser.add_argument("--num_fewshot_examples", type=int, default=3)
    parser.add_argument("--max_prompt_calls", type=int, default=3)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    run(args)
