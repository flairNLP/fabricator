from argparse import ArgumentParser

from datasets import load_dataset

from selection_strategies import select_fewshots
from tam_training import train_classification

task_to_keys = {
    "imdb": {"text_column": ("text", None), "label_column": "label"},
    "rte": {"text_column": ("sentence1", "sentence2"), "label_column": "label"},
    "qnli": {"text_column": ("question", "sentence"), "label_column": "label"},
    "sst2": {"text_column": ("text", None), "label_column": "label"},
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--tam_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--embedding_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--init_strategy", type=str, choices=["random", "closest-to-centeroid", "furthest-to-centeroid", "expected-gradients", "certainty"], default="random")
    parser.add_argument("--stopping_criteria", type=str)
    parser.add_argument("--dataset_size", type=int, default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    args = parser.parse_args()

    full_dataset = load_dataset(args.dataset)
    #TODO stopping_critera = load_stopping_criteria(args.stopping_criteria)
    task_keys = task_to_keys[args.dataset]

    for dataset_size in args.dataset_size:
        dataset = select_fewshots(
            args,
            full_dataset,
            dataset_size,
            task_keys
        )

        train_classification(
            args,
            dataset,
            task_keys
        )
