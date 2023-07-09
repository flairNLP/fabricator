import os
from argparse import ArgumentParser

from datasets import load_dataset
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.dataset_transformations.question_answering import (
    preprocess_squad_format,
    postprocess_squad_format,
)
from ai_dataset_generator.prompts import TextLabelPrompt


def run(arguments):
    """Generate answers based on a few-shot example from context and question."""
    dataset = preprocess_squad_format(load_dataset(arguments.dataset, split=arguments.split))
    fewshot_examples = dataset.select([1, 2, 3])

    prompt = TextLabelPrompt(
        input_variables=arguments.input_variables,
        target_variable=arguments.target_variable,
        task_description=arguments.task_description,
    )

    raw_prompt = prompt.get_prompt_text(fewshot_examples)
    print(raw_prompt)

    unlabeled_examples = dataset.select([10, 20, 30, 40, 50])
    prompt_node = PromptNode(
        model_name_or_path=arguments.llm, api_key=os.environ.get("OPENAI_API_KEY"), max_length=100
    )

    generator = DatasetGenerator(prompt_node)
    generated_dataset, original_dataset = generator.generate(
        support_examples=fewshot_examples,
        unlabeled_examples=unlabeled_examples,
        prompt_template=prompt,
        max_prompt_calls=arguments.max_prompt_calls,
        support_examples_per_prompt=arguments.support_examples_per_prompt,
        return_original_dataset=True,
    )

    generated_dataset = postprocess_squad_format(generated_dataset)

    if arguments.push_to_hub:
        generated_dataset.push_to_hub("your-first-generated-qa-dataset")
        original_dataset.push_to_hub("original-qa-dataset-to-compare")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default="text-davinci-003")
    parser.add_argument("--max_generation_length", type=int, default=100)
    parser.add_argument(
        "--task_description",
        type=str,
        default="Given a context and a question, generate an answer that occurs exactly and only once in the text.",
    )
    parser.add_argument("--dataset", type=str, default="squad_v2")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--input_variables", type=str, nargs="+", default=["context", "question"])
    parser.add_argument("--target_variable", type=str, default="answer")
    parser.add_argument("--output_format", type=str, default="text")
    parser.add_argument("--max_prompt_calls", type=int, default=3)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    run(args)
