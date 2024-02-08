import numpy as np
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, HfArgumentParser
from datasets import load_dataset
import evaluate
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model from huggingface.co/models"}
    )
    tokenizer_name: str = field(
        metadata={"help": "Path to pretrained tokenizer or model from huggingface.co/models"}
    )

def get_influential_subset(dataset):
    # get parameters from config
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_json_file('config.json')

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_imdb = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=2, id2label=id2label, label2id=label2id
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

accuracy = evaluate.load("accuracy")

# example dataset for debugging
imdb = load_dataset("imdb")
get_influential_subset(imdb)

