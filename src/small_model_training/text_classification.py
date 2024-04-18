import math
import torch.nn
import numpy as np
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, HfArgumentParser
from datasets import load_dataset, Dataset
import evaluate
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model from huggingface.co/models"}
    )

def get_influential_subset(dataset):
    id2label = {}
    label2id = {}
    counter = 0

    dataset = Dataset.from_dict(dataset)

    # trace labels
    for entry in dataset:
        if entry['label'] not in label2id:
            label2id[entry['label']] = counter
            id2label[counter] = entry['label']
            counter += 1

    # get parameters from config
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_json_file('config.json')

    # setup preprocessing
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)


    def preprocess_function(batch):
        preprocessed_tokens = tokenizer(batch["text"], truncation=True, padding=True)
        preprocessed_tokens["label"] = [label2id[label] for label in batch["label"]]
        return preprocessed_tokens

    # setup compute_metrics
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    dataset = dataset.train_test_split(test_size=0.2)
    val_test_split = dataset["test"].train_test_split(test_size=0.5)
    dataset["val"] = val_test_split["train"]
    dataset["test"] = val_test_split["test"]

    tokenized_train = dataset["train"].map(preprocess_function, batched=True)
    tokenized_val = dataset["val"].map(preprocess_function, batched=True)
    tokenized_test = dataset["test"].map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=counter, id2label=id2label, label2id=label2id
    )

    # initialize training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    outputs = trainer.predict(tokenized_test)

    inf_subset = []

    indices = calculate_candidates_entropy_knn(outputs, math.ceil(len(dataset["test"])/2))

    for elem in indices:
        inf_subset.append(dataset["test"][elem.item()])

    inf_subset = Dataset.from_list(inf_subset)

    return inf_subset

#TO-DO: dynamic variant for calculating inf_subset
def calculate_candidates_entropy_knn(model_outputs, num_candidates):
    logits = model_outputs[0]

    logits = torch.from_numpy(logits)

    scores = torch.nn.functional.softmax(logits, dim=-1)
    def entropy(probabilities):
        return -torch.sum(probabilities * torch.log2(probabilities))

    entropy_values = [entropy(probabilities) for probabilities in scores]

    sorted_indices = np.argsort(entropy_values)[::-1]

    knn_indices = sorted_indices[:num_candidates]

    return knn_indices
