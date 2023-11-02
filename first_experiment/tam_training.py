import json
from argparse import Namespace

import numpy as np

import evaluate
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

from utils import *


def train_classification(
    args: Namespace,
    dataset: DatasetDict,
    dataset_size: int,
    task_keys: dict
):
    label_column = task_keys["label_column"]
    id2label = dict(enumerate(dataset["train"].features[label_column].names))

    model, tokenizer = get_classification_model_and_tokenizer(args.tam_model, id2label=id2label)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "task_keys": task_keys}
    )

    experiment_extension = (f"{args.tam_model}"
                            f"_{args.dataset}"
                            f"_{dataset_size}"
                            f"_{args.init_strategy}"
                            f"{'_' + args.embedding_model if args.embedding_model is not None else ''}")

    log_path = PATH / experiment_extension

    num_epochs = get_num_epochs(
        batch_size=16,
        dataset_size=len(dataset["train"]),
        max_epochs=5,
        min_steps=200,
    )

    training_args = TrainingArguments(
        output_dir=str(log_path),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        push_to_hub=False,
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.predict(tokenized_dataset["test"])

    with open(log_path / "results.json", "w") as f:
        json.dump(results.metrics, f)
