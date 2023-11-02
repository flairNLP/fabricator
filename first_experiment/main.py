from argparse import ArgumentParser

from datasets import load_dataset

from selection_strategies import select_fewshots
from tam_training import train_classification

task_to_keys = {
    "imdb": {"text_column": ("text", None), "label_column": "label"},
    "rte": {"text_column": ("sentence1", "sentence2"), "label_column": "label"},
    "qnli": {"text_column": ("question", "sentence"), "label_column": "label"},
    "sst2": {"text_column": ("sentence", None), "label_column": "label"},
    "snli": {"text_column": ("premise", "hypothesis"), "label_column": "label"},
}

eval_splits = {
    "imdb": "test",
    "rte": "validation",
    "qnli": "validation",
    "sst2": "validation",
    "snli": "test",
}

dataset_prefixes = {
    "rte": "glue",
    "qnli": "glue",
    "sst2": "glue",
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imdb", choices=["imdb", "rte", "qnli", "sst2", "snli"])
    parser.add_argument("--tam_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--embedding_model", type=str, default=None)
    parser.add_argument("--init_strategy", type=str, choices=["random", "closest-to-centeroid", "furthest-to-centeroid", "expected-gradients", "certainty"], default="random")
    parser.add_argument("--stopping_criteria", type=str)
    parser.add_argument("--dataset_size", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 0])
    args = parser.parse_args()

    if args.dataset in dataset_prefixes:
        full_dataset = load_dataset(dataset_prefixes[args.dataset], args.dataset)
    else:
        full_dataset = load_dataset(args.dataset)
    task_keys = task_to_keys[args.dataset]

    full_dataset["test"] = full_dataset[eval_splits[args.dataset]]

    if args.dataset == "snli":
        full_dataset = full_dataset.filter(lambda x: x["label"] != -1)

    for dataset_size in args.dataset_size:
        if dataset_size > 0:
            dataset = select_fewshots(
                args,
                full_dataset,
                dataset_size,
                task_keys
            )
        else:
            dataset = full_dataset

        train_classification(
            args,
            dataset,
            dataset_size,
            task_keys
        )
