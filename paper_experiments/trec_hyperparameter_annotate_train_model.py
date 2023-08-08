import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import shutil


def run(possible_examples_per_class, fewshot_example_per_class, seed):
    corpus_name = f"whoisjones/trec_hyperparameter_annotated_{possible_examples_per_class}_possible_examples_{fewshot_example_per_class}_used"

    if "corpus_name" not in locals():
        raise Exception("Please insert the generated corpora before running this script.")

    label_alignment = {
        "NUM": "number",
        "ENTY": "entity",
        "DESC": "description",
        "ABBR": "abbreviation",
        "HUM": "human",
        "LOC": "location",
    }
    # Load the dataset
    dataset = load_dataset(corpus_name, split="train").shuffle(seed=seed)
    test_split = load_dataset("trec", split="test")
    original_labels = test_split.features["coarse_label"].names

    def clean_labels(examples):
        label = examples["coarse_label"].replace("Class: ", "")
        if label not in list(label_alignment.values()):
            label = "remove"
        examples["coarse_label"] = label
        return examples

    dataset = dataset.map(clean_labels)
    dataset = dataset.filter(lambda x: x["coarse_label"] != "remove")

    dst_feat = ClassLabel(names=[label_alignment[k] for k in original_labels])
    dataset = dataset.map(lambda batch: {
        "coarse_label": dst_feat.str2int(batch)}, input_columns="coarse_label", batched=True)
    new_features = dataset.features.copy()
    new_features["coarse_label"] = dst_feat
    dataset = dataset.cast(new_features)

    dataset = dataset.train_test_split(test_size=0.1)
    dataset["validation"] = dataset["test"]
    dataset["test"] = test_split
    dataset = dataset.rename_column("coarse_label", "label")
    num_labels = dataset["train"].features["label"].num_classes

    # Load the BERT tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    id2label = dict(enumerate(dataset["train"].features["label"].names))
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to("cuda")

    num_train_epochs = 20

    # Training arguments
    training_args = TrainingArguments(
        output_dir="output_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        evaluation_strategy="epoch",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer.predict(tokenized_dataset["test"])


if __name__ == "__main__":
    # for every combination of possible fewshot examples and fewshot examples used
    for possible_examples_per_class, fewshot_example_per_class in [(0, 0), (2, 1), (2, 2), (4, 1), (4, 2), (4, 3),
                                                                   (4, 4), (8, 1), (8, 2), (8, 3), (8, 4), (16, 1),
                                                                   (16, 2), (16, 3), (16, 4)]:
        result_avg = []
        # iterate over seeds
        for seed in [41, 42, 43, 44, 45]:
            results = run(possible_examples_per_class, fewshot_example_per_class, seed)
            result_avg.append(results.metrics["test_accuracy"] * 100)

        # log for hyperparameter run
        file = f"hyperparameter-trec-annotation-{possible_examples_per_class}-possible-{fewshot_example_per_class}-used"
        with open(f"results/{file}.log", "w") as f:
            f.write(f"Accuracy: {np.mean(result_avg)}\n")
            f.write(f"Standard deviation: {np.std(result_avg)}\n")
