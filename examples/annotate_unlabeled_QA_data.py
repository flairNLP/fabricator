import os
import random
from argparse import ArgumentParser

from datasets import load_dataset
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.prompts import DataGenerationPrompt


def run(args):
    # Preprocess: convert list to str for answer
    def preprocess(example):
        if example["answer"]:
            example["answer"] = example["answer"].pop()
        else:
            example["answer"] = ""
        return example

    dataset = (
        load_dataset(args.dataset, split=args.split).flatten().rename_column("answers.text", "answer").map(preprocess)
    )
    # Select data points that are answerable
    fewshot_examples = dataset.select([50, 55, 60, 65])

    prompt = DataGenerationPrompt(
        input_variables=args.input_variables,
        target_variable=args.target_variable,
        output_format=args.output_format,
        task_description=args.task_description,
    )
    raw_prompt = prompt.get_prompt_text(fewshot_examples)
    print(raw_prompt)

    unlabeled_data = dataset.select([100, 105, 110])
    prompt_node = PromptNode(
        model_name_or_path=args.llm, api_key=os.environ.get("OPENAI_API_KEY"), max_length=args.max_generation_length
    )
    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        support_examples=fewshot_examples,  # from above
        unlabeled_examples=unlabeled_data,
        prompt_template=prompt,  # from above
        max_prompt_calls=args.max_prompt_calls,  # max number of calls to the LLM
        support_examples_per_prompt=args.support_examples_per_prompt,  # number of support examples per prompt
    )

    # Postprocess: calculate the answer start position
    def postprocess(example):
        answer_start = example["context"].find(example["answer"])
        if answer_start < 0:
            print(
                f'Could not calculate the answer start because the context "{example["context"]}" '
                f'does not contain the answer "{example["answer"]}".'
            )
            answer_start = -1
        else:
            # check that the answer doesn't occur more than once in the context
            second_answer_start = example["context"].find(example["answer"], answer_start + 1)
            if second_answer_start >= 0:
                print("Could not calculate the answer start because the context contains the answer more than once.")
                answer_start = -1
            else:
                answer_start = answer_start
        example["answer_start"] = answer_start
        return example

    generated_dataset = generated_dataset.map(postprocess)

    if args.push_to_hub:
        generated_dataset.push_to_hub("your-first-generated-dataset")
    print()


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
    parser.add_argument(
        "--input_variables", type=str, nargs="+", default=["context", "question"]
    )  # Column names as they occur in the dataset
    parser.add_argument("--target_variable", type=str, default="answer")
    parser.add_argument("--output_format", type=str, default="text")  # indicates the output format of the LLM is text
    parser.add_argument("--max_prompt_calls", type=int, default=3)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    run(args)
