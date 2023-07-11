from argparse import ArgumentParser

from evaluation.application_evaluation.application_evaluation import run

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default="text-davinci-003")
    parser.add_argument("--lm", type=str, default="bert-base-uncased")
    parser.add_argument("--max_generation_length", type=int, default=100)
    parser.add_argument("--task_description_generate", type=str, default="Generate similar texts.")
    parser.add_argument(
        "--task_description_annotate",
        type=str,
        default="Classify the review whether it's positive or negative: " "{label_options}.",
    )
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--split_train", type=str, default="train")
    parser.add_argument("--split_test", type=str, default="test")
    # parser.add_argument(
    #    "--input_variables", type=str, nargs="+", default=["text"]
    # )  # Column names as they occur in the dataset
    parser.add_argument(
        "--input_variables", type=str, nargs="+", default=["text"]
    )  # Column names as they occur in the dataset
    parser.add_argument("--num_fewshot_examples", type=int, default=3)
    parser.add_argument("--max_prompt_calls", type=int, default=20)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--target_variable", type=str, default="label")
    parser.add_argument("--torch_device", type=str, default="mps")
    parser.add_argument("--devmode", action="store_true", default=True)
    parser.add_argument("--max_size_generated", type=int, default=1000)
    args = parser.parse_args()
    run(args)
