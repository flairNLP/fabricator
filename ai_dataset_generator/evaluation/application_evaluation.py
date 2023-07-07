import os
import random
from argparse import ArgumentParser
from pathlib import Path

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
from ai_dataset_generator.prompts import GenerateUnlabeledDataPrompt, ClassLabelPrompt


class ApplicationEvaluator:
    """
    Given a dataset (including train and test splits), evaluates the performance of LMs
    trained respectively on:
    * the entire train split
    * various sizes (100, 200, 500, 1000, 2000, etc.) of generated train datasets using
      our dataset generator
    """

    def __init__(self):
        # configuration (change these lines to your needs)
        self.dataset_name = "imdb"
        self.dataset_column_name_text = "text"
        self.dataset_column_name_target = "label"
        self.lm_name = "bert-base-uncased"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # manual overwrite of device if necessary
        # self.device = torch.device("cuda:0")
        # for development: shorten the splits to k rows (disabled if None)
        self.dev_shorten_dataset_splits_to_k_rows = 10
        # pathname of the xlsx file to store the results
        basepath = Path("evaluation/results/")
        basepath.mkdir(parents=True, exist_ok=True)
        self.results_pathname = basepath / f"{self.dataset_name}.xlsx"

        # create pandas dataframe
        self.df = pd.DataFrame()

        # load the dataset
        self.dataset = load_dataset(self.dataset_name)
        logger.info("loaded dataset {}", self.dataset_name)

        # get number of unique labels
        self.num_labels = len(set(self.dataset["train"][self.dataset_column_name_target]))
        logger.info("found {} labels in dataset {}", self.num_labels, self.dataset_name)

        # initialize LM and its tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_name)
        self.lm = AutoModelForSequenceClassification.from_pretrained(self.lm_name, num_labels=self.num_labels).to(
            self.device
        )
        logger.info("initialized LM {}", self.lm_name)

        # tokenize dataset
        self.dataset_tokenized = self.dataset.map(
            ApplicationEvaluator._tokenize_function,
            batched=True,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "dataset_column_name_text": self.dataset_column_name_text,
            },
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        logger.info("tokenized dataset {}", self.dataset_name)

        # shorten dataset splits if necessary
        if self.dev_shorten_dataset_splits_to_k_rows is not None:
            self.dataset_tokenized["train"] = self.dataset_tokenized["train"].select(
                range(self.dev_shorten_dataset_splits_to_k_rows)
            )
            self.dataset_tokenized["test"] = self.dataset_tokenized["test"].select(
                range(self.dev_shorten_dataset_splits_to_k_rows)
            )
            logger.info(
                "DEV: shortened dataset splits to {} rows",
                self.dev_shorten_dataset_splits_to_k_rows,
            )

        # create trainer
        training_args = TrainingArguments("test-trainer", use_mps_device=True)
        trainer = Trainer(
            self.lm,
            training_args,
            train_dataset=self.dataset_tokenized["train"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=ApplicationEvaluator.compute_metrics,
        )

        # train LM on original dataset
        logger.info("training LM on original dataset {}", self.dataset_name)
        trainer.train()
        eval_results = trainer.evaluate(self.dataset_tokenized["test"])
        self.add_evaluation_result(
            "original",
            len(self.dataset_tokenized["train"]),
            len(self.dataset_tokenized["test"]),
            eval_results,
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

        self.df.to_excel(self.results_pathname)
        logger.info("saved {} results to {}", len(self.df), self.results_pathname)


def generate_unlabeled_data(fewshot_examples, arguments):
    prompt = GenerateUnlabeledDataPrompt(
        input_variables=arguments.input_variables,
        task_description=arguments.task_description,
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
    logger.info("generated dataset {}", generated_dataset)

    return generated_dataset


def annotate_dataset(fewshot_examples, generated_unlabeled_dataset, arguments):
    idx2label = dict(enumerate(fewshot_examples.features[arguments.target_variable].names))

    prompt = ClassLabelPrompt(
        input_variables=arguments.input_variables,
        target_variable=arguments.target_variable,
        label_options=idx2label,
        task_description=arguments.task_description,
    )
    raw_prompt = prompt.get_prompt_text(fewshot_examples)
    print(raw_prompt)

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


def run(arguments):
    # evaluator = ApplicationEvaluator()

    # load the dataset split and some few shot examples
    dataset = load_dataset(arguments.dataset, split=arguments.split)
    fewshot_examples = dataset.select(random.sample(range(len(dataset)), arguments.num_fewshot_examples))

    # generate unlabeled dataset using LLM
    generated_unlabeled_dataset = generate_unlabeled_data(fewshot_examples, arguments)

    # annotate unlabeled dataset using LLM
    generated_annotated_dataset = annotate_dataset(fewshot_examples, generated_unlabeled_dataset, arguments)

    print("")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default="text-davinci-003")
    parser.add_argument("--max_generation_length", type=int, default=100)
    parser.add_argument("--task_description", type=str, default="Generate similar texts.")
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--input_variables", type=str, nargs="+", default=["text"]
    )  # Column names as they occur in the dataset
    parser.add_argument("--num_fewshot_examples", type=int, default=3)
    parser.add_argument("--max_prompt_calls", type=int, default=3)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--target_variable", type=str, default="label")
    args = parser.parse_args()
    run(args)
