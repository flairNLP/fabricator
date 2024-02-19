import torch.nn
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer
from datasets import load_dataset
import evaluate

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

accuracy = evaluate.load("accuracy")

model = AutoModelForSequenceClassification.from_pretrained("./imdb_model")

imdb_train = load_dataset("imdb", split="train[:10%]")
imdb_test = load_dataset("imdb", split="test[:1%]")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_imdb_train = imdb_train.map(preprocess_function, batched=True)
tokenized_imdb_test = imdb_test.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_imdb_train,
    eval_dataset=tokenized_imdb_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

outputs = trainer.predict(tokenized_imdb_test)

logits = outputs[0]

logits = torch.from_numpy(logits)

scores = torch.nn.functional.softmax(logits, dim=-1)

first_values = scores[:, 0]
second_values = scores[:, 1]

distance = first_values-second_values

dist = torch.abs(distance)
knn_values, knn_indices = dist.topk(5, largest=False)

fewshot_examples = []

for elem in knn_indices:
    fewshot_examples.append(imdb_test[elem.item()]["text"])

print(fewshot_examples)
