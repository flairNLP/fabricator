import json
from pathlib import Path

import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate

PATH = Path("/glusterfs/dfs-gfs-dist/goldejon/initial-starting-point-generation")

def train_classification(
    args,
    dataset,
    text_column,
    label_column,
):
    id2label = dict(enumerate(dataset["train"].features[label_column].names))
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.tam_model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tam_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if len(text_column) == 1:
        def preprocess_function(examples):
            return tokenizer(examples[text_column[0]], truncation=True)
    else:
        def preprocess_function(examples):
            return tokenizer(examples[text_column[0]], examples[text_column[1]], truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    if args.init_strategy in ["class-centeroid", "class-furthest"]:
        embedding_model = args.embedding_model.split("/")[-1]
        experiment_extension = f"{args.tam_model}_{args.dataset}_{args.dataset_size}_{args.init_strategy}_{embedding_model}"
    else:
        experiment_extension = f"{args.tam_model}_{args.dataset}_{args.dataset_size}_{args.init_strategy}"

    log_path = PATH / experiment_extension

    training_args = TrainingArguments(
        output_dir=str(log_path),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=5,
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
