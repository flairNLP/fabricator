import argparse
import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import evaluate
import shutil


def run(args):
    # iterate over all corpora
    for corpus_name in args.corpora:
        # iterate over all sizes, -1 means we are taking all examples but at most 10k
        for size in [-1, 50, 500, 1000]:
            # Average results for corpus and size over 5 seeds
            result_avg = []
            for seed in [41, 42, 43, 44, 45]:
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

                # preprocess annotated dataset - ensure unified labels (lowercased and no whitespaces) + correct
                # ClassLabel feature
                if "annotated" in corpus_name:
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

                # Compose final training dataset + gold-labeled test split
                if size > 0:
                    dataset = dataset.select(range(size))
                dataset = dataset.train_test_split(test_size=0.1)
                dataset["validation"] = dataset["test"]
                dataset["test"] = test_split
                dataset = dataset.rename_column("coarse_label", "label")
                num_labels = dataset["train"].features["label"].num_classes

                # Load the BERT tokenizer and model
                model_name = "bert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

                # Preprocessing function
                def preprocess_function(examples):
                    return tokenizer(
                        examples["text"],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )

                tokenized_dataset = dataset.map(preprocess_function, batched=True)
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
                accuracy = evaluate.load("accuracy")

                def compute_metrics(eval_pred):
                    predictions, labels = eval_pred
                    predictions = np.argmax(predictions, axis=1)
                    return accuracy.compute(predictions=predictions, references=labels)

                id2label = dict(enumerate(dataset["train"].features["label"].names))
                label2id = {v: k for k, v in id2label.items()}

                # Create model and move to CUDA
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id
                ).to("cuda")

                # Set number of training epochs depending on dataset size
                if size < 0:
                    num_train_epochs = 5
                elif size == 1000:
                    num_train_epochs = 10
                else:
                    num_train_epochs = 20

                # Make tmp path for storing the model
                tmp_path = f"tmp/{corpus_name.replace('/', '-')}-{size}-samples"

                # Training arguments
                training_args = TrainingArguments(
                    output_dir=tmp_path,
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

                results = trainer.predict(tokenized_dataset["test"])
                result_avg.append(results.metrics["test_accuracy"] * 100)

                # remove tmp path since we iterate over seeds, corpora and sizes
                shutil.rmtree(tmp_path)

            # change -1 for logging to 'all'
            if size > 0:
                log_size = str(size)
            else:
                log_size = "all"

            # write results to log file
            log_corpus_name = corpus_name.replace("whoisjones/", "")
            file = f"{log_corpus_name}-{log_size}-samples"
            with open(f"results/{file}.log", "w") as f:
                f.write(f"Accuracy: {np.mean(result_avg)}\n")
                f.write(f"Standard deviation: {np.std(result_avg)}\n")


if __name__ == "__main__":
    # Run like 'python trec_annotation_train_model.py --corpora hfaccount/generated-model trec
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs='+')  # a list of generated and gold-label corpus
    arguments = parser.parse_args()
    run(arguments)
