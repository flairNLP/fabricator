import numpy as np
import datetime
import os
from transformers import AutoTokenizer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, Trainer
from datasets import load_dataset
import evaluate

generated_dataset = load_dataset(
    "stolzenp/500-imdb-movie-reviews-baseline", split="train"
)
test_split = load_dataset("imdb", split="test")

model_name = "bert-base-uncased"

label2id = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
id2label = {1: "positive", 0: "negative"}

# setup preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_text(batch):
    preprocessed_tokens = tokenizer(batch["text"], truncation=True, padding=True)
    return preprocessed_tokens


def preprocess_text_and_labels(batch):
    preprocessed_tokens = tokenizer(batch["text"], truncation=True, padding=True)
    preprocessed_tokens["label"] = [label2id[label] for label in batch["label"]]
    return preprocessed_tokens


# setup compute_metrics
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


generated_dataset = generated_dataset.train_test_split(test_size=0.1)
train_split = generated_dataset["train"]
val_split = generated_dataset["test"]

tokenized_train = train_split.map(preprocess_text_and_labels, batched=True)
tokenized_val = val_split.map(preprocess_text_and_labels, batched=True)
tokenized_test = test_split.map(preprocess_text, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

# initialize training
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="baseline_downstream_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
outputs = trainer.predict(tokenized_test)
test_accuracy = outputs[2]["test_accuracy"]

results_file = "results.txt"
experiments_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = f"{experiments_directory}/{results_file}"

results_timestamp = datetime.datetime.now()

with open(results_path, "a") as file:
    file.write(f"{results_timestamp} - imdb_baseline\n")
    file.write(f"accuracy: {test_accuracy}\n")
