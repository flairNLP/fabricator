from evaluation.application_evaluation.application_evaluation import *


def run_only_annotate(arguments):
    # preprocessing of arguments
    arguments = preprocess_arguments(arguments)

    # get the original dataset and its splits
    dataset_train, dataset_test = get_original_dataset_splits(arguments)

    # train and test the original dataset
    # if arguments.traintest_on_original_dataset:
    #    ApplicationEvaluator(dataset_train, dataset_test, f"original_{len(dataset_train)}", arguments)

    # TODO we could also use our own generated examples as few shot examples in later
    #  iterations in the repeating process below
    # get some few shot examples
    unique_labels = set(dataset_test[arguments.target_variable])
    logger.debug("found {} unique labels: {}", len(unique_labels), unique_labels)
    num_few_shot_examples = arguments.num_fewshot_examples_per_class * len(unique_labels)
    num_few_shot_examples = min(num_few_shot_examples, len(dataset_train))
    logger.info("using {} few shot examples", num_few_shot_examples)
    fewshot_examples = single_label_task_sampler(dataset_train, arguments.target_variable, num_few_shot_examples)
    # indexed_to_select = random.sample(range(len(dataset_train)), num_few_shot_examples)
    # random.shuffle(indexed_to_select)
    # logger.debug("selecting examples with indexes {}", indexed_to_select)
    # fewshot_examples = dataset_train.select(indexed_to_select)
    labels = fewshot_examples["label"]
    logger.info("few shot examples label distribution: {}", Counter(labels))

    _, label_options = convert_label_ids_to_texts(
        fewshot_examples,
        arguments.target_variable,
        expanded_label_mapping=arguments.label2human_friendly_label,
        return_label_options=True,
    )

    # 1) generate a dataset and annotate it
    # 2) extend the overall generated dataset with the generated dataset from step 1)
    # 3) train and test the generated dataset
    # 4) repeat until the generated dataset has reached the desired size
    generated_annotated_dataset = None
    for current_desired_size in (500, 1000):
        # get a subset of the original dataset of the desired size
        dataset_train_subset = dataset_train.select(range(current_desired_size))

        # train and test the original dataset of same size if requested
        if arguments.traintest_on_original_dataset:
            logger.info("training and testing of original dataset of same size...")

            ApplicationEvaluator(
                dataset_train_subset, dataset_test, f"original_{len(dataset_train_subset)}", arguments
            )
            logger.info("finished training and testing of original dataset of same size")

        # now remove the target column and replace it with placeholder values
        dataset_train_subset = dataset_train_subset.remove_columns([arguments.target_variable])
        dataset_train_subset = add_placeholder_labels(fewshot_examples, dataset_train_subset, arguments)

        # annotate a dataset using the power of LLM
        current_generated_annotated_dataset = annotate_dataset(
            fewshot_examples, dataset_train_subset, arguments, label_options
        )

        # save to disk (will be overwritten each time)
        filepath = (
            DATASETPATH
            / f"current_generated_annotated_dataset_{arguments.dataset}_{len(current_generated_annotated_dataset)}.xlsx"
        )
        current_generated_annotated_dataset.to_pandas().to_excel(filepath)

        # extend the overall dataset with the newly generated one
        if generated_annotated_dataset is None:
            generated_annotated_dataset = current_generated_annotated_dataset
        else:
            generated_annotated_dataset = datasets.concatenate_datasets(
                [generated_annotated_dataset, current_generated_annotated_dataset]
            )
        texts = generated_annotated_dataset["text"]
        labels = generated_annotated_dataset["label"]
        logger.info("extended generated dataset to size {}", len(generated_annotated_dataset))
        logger.debug("label distribution: {}", Counter(labels))

        # save the generated dataset to disk
        filepath = DATASETPATH / f"generated_annotated_dataset_{arguments.dataset}_{len(generated_annotated_dataset)}"
        generated_annotated_dataset.save_to_disk(filepath)
        generated_annotated_dataset.to_pandas().to_excel(str(filepath) + ".xlsx")

        # train and test the generated dataset
        logger.info("training and testing of generated dataset...")
        ApplicationEvaluator(
            generated_annotated_dataset, dataset_test, f"generated_{len(generated_annotated_dataset)}", arguments
        )
        logger.info("finished training and testing of generated dataset")

        logger.info("finished iteration of dataset generation with current size {}", len(generated_annotated_dataset))


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
