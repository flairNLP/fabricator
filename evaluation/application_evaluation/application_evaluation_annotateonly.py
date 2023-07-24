from ai_dataset_generator.samplers.samplers import single_label_stratified_sample
from evaluation.application_evaluation.application_evaluation import *


def run_only_annotate(arguments):
    # preprocessing of arguments
    arguments = preprocess_arguments(arguments)

    # get the original dataset and its splits
    dataset_train, dataset_test = get_original_dataset_splits(arguments)

    # get some few shot examples
    unique_labels = set(dataset_test[arguments.target_variable])
    logger.debug("found {} unique labels: {}", len(unique_labels), unique_labels)
    num_few_shot_examples = arguments.num_fewshot_examples_per_class * len(unique_labels)
    num_few_shot_examples = min(num_few_shot_examples, len(dataset_train))
    logger.info("using {} few shot examples", num_few_shot_examples)
    fewshot_examples, dataset_train_unused = single_label_stratified_sample(dataset_train, arguments.target_variable, arguments.num_fewshot_examples_per_class, return_unused_split=True)
    labels = fewshot_examples["label"]
    logger.info("few shot examples label distribution: {}", Counter(labels))
    # save to disk for reproducibility
    filepath = DATASETPATH / f"{arguments.dataset}_fewshot_{len(fewshot_examples)}"
    fewshot_examples.save_to_disk(filepath)

    # convert
    fewshot_examples, label_options = convert_label_ids_to_texts(
        fewshot_examples,
        arguments.target_variable,
        expanded_label_mapping=arguments.label2human_friendly_label,
        return_label_options=True,
    )

    # get a subset of the training data that we will then annotate
    dataset_train_subset = single_label_stratified_sample(dataset_train_unused, arguments.target_variable, int(arguments.max_prompt_calls / len(unique_labels)))
    labels = dataset_train_subset["label"]
    logger.info("subset train label distribution: {}", Counter(labels))
    # save to disk for reproducibility
    filepath = DATASETPATH / f"{arguments.dataset}_trainsubset_{len(dataset_train_subset)}"
    dataset_train_subset.save_to_disk(filepath)

    # now remove the target column and replace it with placeholder values
    dataset_train_subset_nolabels = dataset_train_subset.remove_columns([arguments.target_variable])
    dataset_train_subset_nolabels = add_placeholder_labels(fewshot_examples, dataset_train_subset_nolabels, arguments)
    # save to disk for reproducibility
    filepath = DATASETPATH / f"{arguments.dataset}_trainsubset_labelsremoved_{len(dataset_train_subset_nolabels)}"
    dataset_train_subset_nolabels.save_to_disk(filepath)

    # annotate a dataset using the power of LLM
    dataset_train_subset_nolabels_autolabeled = annotate_dataset(
        fewshot_examples, dataset_train_subset_nolabels, arguments, label_options
    )
    dataset_train_subset_nolabels_autolabeled = dataset_train_subset_nolabels_autolabeled.class_encode_column(arguments.target_variable)
    # save to disk (will be overwritten each time)
    filepath = DATASETPATH / f"{arguments.dataset}_trainsubset_labelsremoved_autolabeled_{len(dataset_train_subset_nolabels_autolabeled)}"
    dataset_train_subset_nolabels_autolabeled.save_to_disk(filepath)
    filepath_xlsx = DATASETPATH / f"{arguments.dataset}_trainsubset_labelsremoved_autolabeled_{len(dataset_train_subset_nolabels_autolabeled)}.xlsx"
    dataset_train_subset_nolabels_autolabeled.to_pandas().to_excel(filepath_xlsx)


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
    parser.add_argument("--max_prompt_calls", type=int, default=10000)
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
    run_only_annotate(args)
