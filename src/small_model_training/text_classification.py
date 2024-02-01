import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import json
import evaluate

def get_influential_subset(dataset):
    # get parameters from dict
    data = get_training_parameters()
    small_model = data['small_model']
    batch_size = data['batch_size']

    tokenized_imdb = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


    # TO-DO: calculate influential dataset
    inf_subset = dataset

    # TO-DO: check for pre-processing
    return inf_subset

def get_training_parameters():

    # open config file
    f = open('training_parameters.json')

    # return json object as dict
    data = json.load(f)

    # close file
    f.close()
    return data

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# example dataset for debugging
imdb = load_dataset("imdb")
get_influential_subset(imdb)

