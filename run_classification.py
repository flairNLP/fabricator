from argparse import ArgumentParser

from datasets import load_dataset, Dataset

from selection_strategies import select_fewshots
from training import train_classification

column_mapping = {
    "imdb": {"text_column": ["text"], "label_column": "label"},
}


def run(args):
    full_dataset = load_dataset(args.dataset)
    text_column = column_mapping[args.dataset]["text_column"]
    label_column = column_mapping[args.dataset]["label_column"]

    dataset = select_fewshots(
        args,
        full_dataset,
        text_column,
        label_column,
    )

    train_classification(
        args,
        dataset,
        text_column,
        label_column,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--tam_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--embedding_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--init_strategy", type=str, choices=["random", "class-centeroid-closest", "class-centeroid-furthest", "expected-gradients", "certainty"], default="random")
    parser.add_argument("--dataset_size", type=int, default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    arguments = parser.parse_args()
    run(arguments)