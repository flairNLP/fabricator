from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default="text-davinci-003")
    parser.add_argument("--max_generation_length", type=int, default=100)
    parser.add_argument("--task_description", type=str, default="Generate similar texts.")
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--input_variables", type=str, nargs="+", default=["text"]
    )  # Column names as they occur in the dataset
    parser.add_argument("--output_format", type=str, default="text")  # indicates the output format of the LLM is text
    parser.add_argument("--num_fewshot_examples", type=int, default=3)
    parser.add_argument("--max_prompt_calls", type=int, default=3)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()
