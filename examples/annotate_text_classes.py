import os
import random
from argparse import ArgumentParser

from datasets import load_dataset
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.prompts import infer_prompt_from_dataset


def run(arguments):
    """Generate annotations for unlabeled data for a given dataset and split."""
    dataset = load_dataset(arguments.dataset, split=arguments.split)
    fewshot_examples = dataset.select(random.sample(range(len(dataset)), arguments.num_fewshot_examples))

    prompt = infer_prompt_from_dataset(dataset)

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

    if arguments.push_to_hub:
        generated_dataset.push_to_hub("your-first-generated-dataset")
    print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default="text-davinci-003")
    parser.add_argument("--max_generation_length", type=int, default=100)
    parser.add_argument(
        "--task_description",
        type=str,
        default="Classify the review whether it's positive or negative: " "{label_options}.",
    )
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--input_columns", type=str, nargs="+", default=["text"]
    )  # Column names as they occur in the dataset
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument(
        "--output_format", type=str, default="single_label_classification"
    )  # indicates the output format of the LLM is text
    parser.add_argument("--num_fewshot_examples", type=int, default=3)
    parser.add_argument("--max_prompt_calls", type=int, default=3)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    run(args)
