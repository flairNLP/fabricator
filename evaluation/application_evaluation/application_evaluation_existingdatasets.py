from datasets import Dataset

from evaluation.application_evaluation.application_evaluation import *


def log_dataset_statistics(dataset):
    logger.info("dataset size: {}", len(dataset))
    logger.info("dataset label distribution: {}", Counter(dataset["label"]))


def replace_values(example):
    if example["label"] == 3:
        example["label"] = 0
    elif example["label"] == 2:
        example["label"] = 0
    return example


def run(arguments):
    path_autolabeled = DATASETPATH / "imdb_trainsubset_labelsremoved_autolabeled_10000"
    path_gold = DATASETPATH / "imdb_trainsubset_10000"

    dataset_autolabeled = Dataset.load_from_disk(path_autolabeled).shuffle()
    dataset_autolabeled = dataset_autolabeled.map(replace_values)
    logger.info("autolabeled:")
    log_dataset_statistics(dataset_autolabeled)

    dataset_gold = Dataset.load_from_disk(path_gold).shuffle()
    logger.info("gold:")
    log_dataset_statistics(dataset_gold)

    dataset_test = load_dataset("imdb", split="test")
    logger.info("test:")
    log_dataset_statistics(dataset_test)

    for current_size in (500, 1000, 2000, 5000, 10000):
        # get subset
        current_train_autolabeled = dataset_autolabeled.select(range(current_size))
        current_train_gold = dataset_autolabeled.select(range(current_size))

        # train and test
        ApplicationEvaluator(current_train_autolabeled, dataset_test, "autolabeled", arguments)
        ApplicationEvaluator(current_train_gold, dataset_test, "gold", arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--lm", type=str, default="bert-base-uncased")
    parser.add_argument("--max_generation_length", type=int, default=500)
    parser.add_argument(
        "--task_description_generate", type=str, default=GenerateUnlabeledDataPrompt.DEFAULT_TASK_DESCRIPTION
    )
    parser.add_argument("--task_description_annotate", type=str, default=ClassLabelPrompt.DEFAULT_TASK_DESCRIPTION)
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--split_train", type=str, default="train")
    parser.add_argument("--split_test", type=str, default="test")
    parser.add_argument("--input_variables", type=str, nargs="+", default=["text"])
    parser.add_argument("--num_fewshot_examples_per_class", type=int, default=2)
    parser.add_argument("--max_prompt_calls", type=int, default=50)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--target_variable", type=str, default="label")
    parser.add_argument("--torch_device", type=str, default="cuda")
    parser.add_argument("--devmode", action="store_true", default=False)
    parser.add_argument("--max_size_generated", type=int, default=200)
    parser.add_argument("--traintest_on_original_dataset", action="store_true", default=False)
    parser.add_argument("--l2hfl", action="append", type=lambda kv: kv.split("="), dest="label2human_friendly_label")
    parser.add_argument(
        "--hfl2d", action="append", type=lambda kv: kv.split("="), dest="human_friendly_label2description"
    )
    args = parser.parse_args()
    args = preprocess_arguments(args)
    run(args)
