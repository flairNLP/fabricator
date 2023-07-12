import argparse
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from haystack.nodes import PromptNode
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from transformers import TrainingArguments, Trainer

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.dataset_transformations.text_classification import \
    convert_label_ids_to_texts
from ai_dataset_generator.prompts import GenerateUnlabeledDataPrompt, ClassLabelPrompt

BASEPATH = Path("evaluation/application_evaluation")
RESULTSPATH = BASEPATH / "results"
DATASETPATH = BASEPATH / "datasets"
BASEPATH.mkdir(parents=True, exist_ok=True)
RESULTSPATH.mkdir(parents=True, exist_ok=True)
DATASETPATH.mkdir(parents=True, exist_ok=True)


class ApplicationEvaluator:
    """
    Given a dataset (including train and test splits), evaluates the performance of LMs
    trained respectively on:
    * the entire train split
    * various sizes (100, 200, 500, 1000, 2000, etc.) of generated train datasets using
      our dataset generator
    """

    def __init__(self, dataset_train: datasets.Dataset, dataset_test: datasets.Dataset, run_type: str, arguments):
        assert len(arguments.input_variables) == 1, "currently only 1 input variable is supported"
        self.dataset_column_name_text = arguments.input_variables[0]
        self.dataset_column_name_target = arguments.target_variable
        self.lm_name = arguments.lm
        self.dataset_name = arguments.dataset
        self.device = torch.device(arguments.torch_device)
        if arguments.devmode:
            # for development: shorten the splits to k rows (disabled if None)
            self.dev_shorten_dataset_splits_to_k_rows = 10
        else:
            self.dev_shorten_dataset_splits_to_k_rows = None
        # pathname of the xlsx file to store the results

        self.results_pathname = RESULTSPATH / f"{self.dataset_name}.xlsx"
        self.run_type = run_type

        # create pandas dataframe or use existing one from disk
        try:
            self.df = pd.read_excel(self.results_pathname)
            logger.info("read {} results from {}", len(self.df), self.results_pathname)
        except FileNotFoundError:
            self.df = pd.DataFrame()
            logger.info("created new results dataframe")

        # datasets
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test

        # get number of unique labels
        unique_labels = set(self.dataset_test[self.dataset_column_name_target])
        self.num_labels = len(unique_labels)
        logger.info("found {} labels in dataset {}: {}", self.num_labels, self.dataset_name, unique_labels)

        # initialize LM and its tokenizer
        logger.debug("using device {}", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_name)
        self.lm = AutoModelForSequenceClassification.from_pretrained(self.lm_name, num_labels=self.num_labels).to(
            self.device
        )
        logger.info("initialized LM {}", self.lm_name)

        # shorten dataset splits if necessary
        if self.dev_shorten_dataset_splits_to_k_rows is not None:
            self.dataset_train = self.dataset_train.select(
                range(min(self.dev_shorten_dataset_splits_to_k_rows, len(self.dataset_train)))
            )
            self.dataset_test = self.dataset_test.select(
                range(min(self.dev_shorten_dataset_splits_to_k_rows, len(self.dataset_test)))
            )
            logger.error(
                "--- !!! DEV !!!: shortened datasets to {} rows !!! ---",
                self.dev_shorten_dataset_splits_to_k_rows,
            )

        # tokenize datasets
        self.dataset_train_tokenized = self.tokenize_dataset(self.dataset_train)
        self.dataset_test_tokenized = self.tokenize_dataset(self.dataset_test)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        logger.info("tokenized datasets", self.dataset_name)

        # create trainer
        training_args = TrainingArguments("trainer")
        trainer = Trainer(
            self.lm,
            training_args,
            train_dataset=self.dataset_train_tokenized,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=ApplicationEvaluator.compute_metrics,
        )

        # train LM on original dataset
        logger.info("training LM on dataset with size {}", len(self.dataset_train_tokenized))
        trainer.train()
        eval_results = trainer.evaluate(self.dataset_test_tokenized)
        self.add_evaluation_result(
            self.run_type,
            len(self.dataset_train_tokenized),
            len(self.dataset_test_tokenized),
            eval_results,
        )

    def tokenize_dataset(self, dataset):
        return dataset.map(
            ApplicationEvaluator._tokenize_function,
            batched=True,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "dataset_column_name_text": self.dataset_column_name_text,
            },
        )

    @staticmethod
    def _tokenize_function(example, tokenizer, dataset_column_name_text):
        return tokenizer(example[dataset_column_name_text], truncation=True)

    @staticmethod
    def compute_metrics(eval_preds):
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels),
            "f1": f1.compute(predictions=predictions, references=labels),
            "precision": precision.compute(predictions=predictions, references=labels),
            "recall": recall.compute(predictions=predictions, references=labels),
        }

    def add_evaluation_result(
        self,
        run_type: str,
        training_size: int,
        test_size: int,
        eval_results: dict,
    ):
        """
        Adds an evaluation result to the df and stores the df to disk as an Excel file.
        :param run_type:
        :param dataset_name:
        :param training_size:
        :param test_size:
        :param eval_results:
        :return:
        """
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    [
                        {
                            "dt_utc": datetime.utcnow(),
                            "run_type": run_type,
                            "training_size": training_size,
                            "test_size": test_size,
                            "accuracy": eval_results["eval_accuracy"]["accuracy"],
                            "f1": eval_results["eval_f1"]["f1"],
                            "precision": eval_results["eval_precision"]["precision"],
                            "recall": eval_results["eval_recall"]["recall"],
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

        self.df.to_excel(self.results_pathname, index=False)
        logger.info("saved {} results to {}", len(self.df), self.results_pathname.resolve())


def generate_unlabeled_data(fewshot_examples, arguments):
    prompt = GenerateUnlabeledDataPrompt(
        input_variables=arguments.input_variables,
        task_description=arguments.task_description_generate,
    )

    prompt_node = PromptNode(
        model_name_or_path=arguments.llm,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_length=arguments.max_generation_length,
    )
    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        support_examples=fewshot_examples,
        prompt_template=prompt,
        max_prompt_calls=arguments.max_prompt_calls,
        support_examples_per_prompt=arguments.support_examples_per_prompt,
    )

    # add the empty target column with proper column type
    original_dtype = fewshot_examples.features[arguments.target_variable].dtype
    if original_dtype in ("float32", "float64"):
        placeholder_value = -100000.0
    elif original_dtype in ("int32", "int64"):
        placeholder_value = -100000
    elif original_dtype == "string":
        placeholder_value = "-100000"
    else:
        raise ValueError("unsupported dtype {} - please define a placeholder value", original_dtype)
    new_column = [placeholder_value] * len(generated_dataset)
    generated_dataset = generated_dataset.add_column(arguments.target_variable, new_column)

    logger.info("generated dataset {}", generated_dataset)

    return generated_dataset


def annotate_dataset(fewshot_examples, generated_unlabeled_dataset, arguments, label_options=None):
    # if no label options are provided, generate them adhoc from the dataset
    if label_options is None:
        label_options = dict(enumerate(fewshot_examples.features[arguments.target_variable].names))

    task_description = arguments.task_description_annotate

    # if a description for each label is described, add it to the task description
    if arguments.human_friendly_label2description:
        label_explanations = "\n".join([f"{k}: {v}" for k, v in arguments.human_friendly_label2description.items()])
        task_description = task_description + "\n\nEach label is described as follows:\n" + label_explanations

    prompt = ClassLabelPrompt(
        input_variables=arguments.input_variables,
        target_variable=arguments.target_variable,
        label_options=label_options,
        task_description=task_description,
    )

    prompt_node = PromptNode(
        model_name_or_path=arguments.llm,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_length=arguments.max_generation_length,
    )
    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        support_examples=fewshot_examples,  # from above
        unlabeled_examples=generated_unlabeled_dataset,
        prompt_template=prompt,  # from above
        max_prompt_calls=arguments.max_prompt_calls,  # max number of calls to the LLM
        support_examples_per_prompt=arguments.support_examples_per_prompt,  # number of support examples per prompt
    )
    logger.info("annotated dataset {}", generated_dataset)

    return generated_dataset


def get_original_dataset_splits(arguments):
    # load the dataset
    dataset = load_dataset(arguments.dataset)
    logger.info("loaded dataset {}", arguments.dataset)

    return dataset[arguments.split_train].shuffle(), dataset[arguments.split_test].shuffle()


def generate_and_annotate_dataset(fewshot_examples, arguments):
    fewshot_examples, label_options = convert_label_ids_to_texts(
        fewshot_examples,
        arguments.target_variable,
        expanded_label_mapping=arguments.label2human_friendly_label,
        return_label_options=True,
    )

    # generate unlabeled dataset using LLM
    generated_unlabeled_dataset = generate_unlabeled_data(fewshot_examples, arguments)

    # annotate unlabeled dataset using LLM
    generated_annotated_dataset = annotate_dataset(
        fewshot_examples, generated_unlabeled_dataset, arguments, label_options
    )

    generated_annotated_dataset = generated_annotated_dataset.filter(lambda example: example[arguments.target_variable] in label_options)
    junk_labeled_dataset = generated_annotated_dataset.filter(lambda example: example[arguments.target_variable] not in label_options)
    if junk_labeled_dataset:
        logger.warning("found {} examples with junk labels", len(junk_labeled_dataset))
        logger.warning("saving junk labeled dataset to disk.")
        filepath = DATASETPATH / f"junk_labeled_dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        junk_labeled_dataset.save_to_disk(filepath)

    generated_annotated_dataset = generated_annotated_dataset.class_encode_column(arguments.target_variable)

    texts = generated_annotated_dataset["text"]
    labels = generated_annotated_dataset["label"]
    logger.info("current generated dataset size {}", len(generated_annotated_dataset))
    logger.debug("label distribution: {}", Counter(labels))

    return generated_annotated_dataset


def preprocess_arguments(arguments):
    arg_dict = vars(arguments)

    expanded_label_mapping = {
        "0": "negative",
        "1": "positive",
    }
    arg_dict["label2human_friendly_label"] = expanded_label_mapping
    one_sentence_description = {
        "negative": "A negative movie review.",
        "positive": "A positive movie review.",
    }
    arg_dict["human_friendly_label2description"] = one_sentence_description

    return argparse.Namespace(**arg_dict)


def run(arguments):
    # preprocessing of arguments
    arguments = preprocess_arguments(arguments)

    # get the original dataset and its splits
    dataset_train, dataset_test = get_original_dataset_splits(arguments)

    # train and test the original dataset
    if arguments.traintest_on_original_dataset:
        ApplicationEvaluator(dataset_train, dataset_test, f"original_{len(dataset_train)}", arguments)

    # TODO we could also use our own generated examples as few shot examples in later
    #  iterations in the repeating process below
    # get some few shot examples
    unique_labels = set(dataset_test[arguments.target_variable])
    logger.debug("found {} unique labels: {}", len(unique_labels), unique_labels)
    num_few_shot_examples = arguments.num_fewshot_examples_per_class * len(unique_labels)
    num_few_shot_examples = min(num_few_shot_examples, len(dataset_train))
    logger.info("using {} few shot examples", num_few_shot_examples)
    indexed_to_select = random.sample(range(len(dataset_train)), num_few_shot_examples)
    random.shuffle(indexed_to_select)
    logger.debug("selecting examples with indexes {}", indexed_to_select)
    fewshot_examples = dataset_train.select(indexed_to_select)
    labels = fewshot_examples["label"]

    # 1) generate a dataset and annotate it
    # 2) extend the overall generated dataset with the generated dataset from step 1)
    # 3) train and test the generated dataset
    # 4) repeat until the generated dataset has reached the desired size
    generated_annotated_dataset = None
    while generated_annotated_dataset is None or len(generated_annotated_dataset) < arguments.max_size_generated:
        if arguments.devmode:
            # pretend we are generating and annotating a new dataset by always loading
            # the same one from disk to save money (@Jonas: love you buddy)
            try:
                current_generated_annotated_dataset = datasets.load_from_disk(
                    DATASETPATH / "generated_annotated_dataset_imdb_20_dev.dataset"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "please run the script with --devmode=False once to generate the dataset, then rename it match the filename above"
                )
        else:
            # generate a dataset and annotate it using the power of LLM
            current_generated_annotated_dataset = generate_and_annotate_dataset(fewshot_examples, arguments)

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
        ApplicationEvaluator(
            generated_annotated_dataset, dataset_test, f"generated_{len(generated_annotated_dataset)}", arguments
        )
        # train and test the original dataset of same size if requested
        if arguments.traintest_on_original_dataset:
            dataset_train_subset = dataset_train.select(range(len(generated_annotated_dataset)))
            ApplicationEvaluator(
                dataset_train_subset, dataset_test, f"original_{len(dataset_train_subset)}", arguments
            )


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
    parser.add_argument("--max_prompt_calls", type=int, default=10)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--target_variable", type=str, default="label")
    parser.add_argument("--torch_device", type=str, default="cuda")
    parser.add_argument("--devmode", action="store_true", default=False)
    parser.add_argument("--max_size_generated", type=int, default=200)
    parser.add_argument("--traintest_on_original_dataset", action="store_true", default=False)
    parser.add_argument("--l2hfl", action="append", type=lambda kv: kv.split("="), dest="label2humanfriendlylabel")
    parser.add_argument(
        "--hfl2d", action="append", type=lambda kv: kv.split("="), dest="humanfriendlylabel2description"
    )
    args = parser.parse_args()
    args = preprocess_arguments(args)
    run(args)
