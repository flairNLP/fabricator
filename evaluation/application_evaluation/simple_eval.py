from datasets import load_from_disk
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

from evaluation.application_evaluation.application_evaluation import DATASETPATH
from evaluation.application_evaluation.application_evaluation_existingdatasets import \
    log_dataset_statistics, replace_values


def run():
    logger.info("Loading datasets")
    dataset_gold = load_from_disk(DATASETPATH / "imdb_trainsubset_10000")
    dataset_autolabeled = load_from_disk(DATASETPATH / "imdb_trainsubset_labelsremoved_autolabeled_10000")
    logger.info("autolabeled:")
    log_dataset_statistics(dataset_autolabeled)
    dataset_autolabeled = dataset_autolabeled.map(replace_values, load_from_cache_file=False)
    logger.info("autolabeled:")
    log_dataset_statistics(dataset_autolabeled)
    logger.info("gold:")
    log_dataset_statistics(dataset_gold)

    logger.info("extracting target column")
    texts1 = list(dataset_gold["text"])
    texts2 = list(dataset_autolabeled["text"])
    targets_gold = list(dataset_gold["label"])
    targets_auto = list(dataset_autolabeled["label"])

    logger.info("same size check...")
    assert len(targets_gold) == len(targets_auto), "Targets are not of the same size"

    logger.info("same order check...")

    for index in tqdm(range(len(texts1))):
        assert texts1[index] == texts2[index]
    logger.info("datasets are of same size and in same order")

    # Compute the metrics
    precision = precision_score(targets_gold, targets_auto, average="macro")  # Adjust according to your problem
    recall = recall_score(targets_gold, targets_auto, average="macro")  # Adjust according to your problem
    f1 = f1_score(targets_gold, targets_auto, average="macro")  # Adjust according to your problem
    accuracy = accuracy_score(targets_gold, targets_auto)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    run()
