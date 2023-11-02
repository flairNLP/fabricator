import os
import json
from collections import Counter
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


def select_fewshots(args, full_dataset, text_column, label_column) -> Dataset:
    if args.init_strategy == "random":
        dataset = random_selection(full_dataset, args.dataset_size, label_column)
    elif args.init_strategy == "class-centeroid-closest":
        dataset = closest_to_centeroid_selection(args.embedding_model, full_dataset, args.dataset_size, text_column,
                                                 label_column)
    elif args.init_strategy == "class-centeroid-furthest":
        dataset = furthest_to_centeroid_selection(args.embedding_model, full_dataset, args.dataset_size, text_column,
                                                  label_column)
    elif args.init_strategy == "expected-gradients":
        dataset = expected_gradients_selection(args.tam_model, full_dataset, args.dataset_size, text_column,
                                               label_column)
    elif args.init_strategy == "certainty":
        dataset = entropy_selection(args.tam_model, full_dataset, args.dataset_size, text_column, label_column)
    else:
        raise NotImplementedError

    return dataset


def random_selection(dataset, num_total_samples, label_column) -> Dataset:
    dataset = dataset.shuffle(seed=42)
    id2label = dict(enumerate(dataset["train"].features[label_column].names))
    num_samples_per_class = num_total_samples // len(dataset["train"].features[label_column].names)
    counter = Counter({idx: 0 for idx in id2label.keys()})
    selected_examples = []
    for idx, example in enumerate(dataset["train"]):
        if counter.get(example[label_column]) < num_samples_per_class:
            counter[example[label_column]] += 1
            selected_examples.append(idx)
            continue
        if all([count == num_samples_per_class for count in counter.values()]):
            break
    dataset["train"] = dataset["train"].select(selected_examples)
    return dataset


def closest_to_centeroid_selection(
    model_name_or_path,
    dataset,
    num_total_samples,
    text_column,
    label_column,
) -> Dataset:
    return selection_with_class_centeroids(
        model_name_or_path,
        dataset,
        num_total_samples,
        text_column,
        label_column,
        selection_strategy="closest"
    )

def furthest_to_centeroid_selection(
    model_name_or_path,
    dataset,
    num_total_samples,
    text_column,
    label_column,
) -> Dataset:
    return selection_with_class_centeroids(
        model_name_or_path,
        dataset,
        num_total_samples,
        text_column,
        label_column,
        selection_strategy="furthest"
    )


def selection_with_class_centeroids(
        model_name_or_path,
        dataset,
        num_total_samples,
        text_column,
        label_column,
        selection_strategy,
) -> Dataset:
    if f"{dataset}-centeroid-{selection_strategy}.json" in os.listdir(".cache/"):
        distance_to_center = json.load(f".cache/{dataset}-centeroid-{selection_strategy}.json")
    else:
        model = AutoModel.from_pretrained(model_name_or_path)
        if torch.cuda.is_available():
            model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if len(text_column) == 1:
            loader = DataLoader(dataset["train"][text_column[0]], batch_size=32, shuffle=False)
        else:
            raise NotImplementedError

        embeddings = []

        for batch in tqdm(loader):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
            embeddings.extend(outputs.last_hidden_state[:, 0, :].detach().cpu().numpy())

        labels = [(i, l) for i, l in enumerate(dataset['train'][label_column])]
        id2label = dict(enumerate(dataset["train"].features[label_column].names))
        distance_to_center = []

        def cosine(x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        for label_idx in id2label.keys():
            class_embeddings = [emb for label, emb in zip(labels, embeddings) if label[1] == label_idx]
            class_centeroid = np.mean(class_embeddings, axis=0)
            dists = [cosine(class_centeroid, emb) for emb in class_embeddings]
            class_labels = [label for label in labels if label[1] == label_idx]
            distance_to_center.extend([(i, l, d) for (i, l), d in zip(class_labels, dists)])

        with open(f".cache/{dataset}-centeroid-{selection_strategy}.json") as f:
            json.dump(distance_to_center, f)

    id2label = dict(enumerate(dataset["train"].features[label_column].names))
    num_examples_per_class = num_total_samples // len(id2label.keys())
    selected_examples = []
    for label_idx in id2label.keys():
        class_distance_to_center = [dist_tuple for dist_tuple in distance_to_center if dist_tuple[1] == label_idx]
        sorted_di = sorted(class_distance_to_center, key=lambda x: x[2], reverse=True if selection_strategy == "closest" else False)
        selected_examples.extend([idx for idx, _, _ in sorted_di[:num_examples_per_class]])

    dataset["train"] = dataset["train"].select(selected_examples)
    return dataset

def expected_gradients_selection(
    model_name_or_path,
    dataset,
    num_total_samples,
    text_column,
    label_column,
) -> Dataset:
    if f"{dataset}-expected-gradients.json" in os.listdir(".cache/"):
        expected_gradients = json.load(f".cache/{dataset}-expected-gradients.json")

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

        if torch.cuda.is_available():
            model.cuda()

        if len(text_column) == 1:
            loader = DataLoader(dataset["train"][text_column[0]], batch_size=8, shuffle=False)
        else:
            raise NotImplementedError

        expected_gradients = []

        params_filter = [n for n, p in model.named_parameters() if not p.requires_grad]

        model.train()
        example_id = 0
        for idx, batch in tqdm(enumerate(loader)):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
            model.zero_grad()
            criterion = torch.nn.CrossEntropyLoss(reduction="none")
            outputs = model(**inputs)
            targets = torch.tensor(dataset["train"][label_column][idx * batch_size: (idx + 1) * batch_size]).to(model.device)
            loss = criterion(outputs.logits, targets)
            for single_loss, single_target in zip(loss, targets):
                gradients = torch.autograd.grad(
                    outputs=single_loss,
                    inputs=[
                        param for name, param
                        in model.named_parameters()
                        if name not in params_filter],
                    create_graph=True)
                gradients = [a.detach() for a in gradients]
                expected_gradients.append((
                    example_id,
                    single_target.cpu().numpy().tolist(),
                    torch.norm(torch.cat([layer_grad.view(-1) for layer_grad in gradients]), p=2).cpu().numpy().tolist()
                    ))
                example_id += 1

        with open(f".cache/{dataset}-expected-gradients.json") as f:
            json.dump(expected_gradients, f)

    id2label = dict(enumerate(dataset["train"].features[label_column].names))
    num_examples_per_class = num_total_samples // len(id2label.keys())
    selected_examples = []

    for i, label in id2label.items():
        class_expected_gradients = [grad_tuple for grad_tuple in expected_gradients if grad_tuple[1] == i]
        sorted_class_expected_gradients = sorted(class_expected_gradients, key=lambda x: x[2], reverse=True)
        selected_examples.extend([grad_tuple[0] for grad_tuple in sorted_class_expected_gradients[:num_examples_per_class]])

    dataset["train"] = dataset["train"].select(selected_examples)
    return dataset

def entropy_selection(
    model_name_or_path,
    dataset,
    num_total_samples,
    text_column,
    label_column,
):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if torch.cuda.is_available():
        model.cuda()

    if len(text_column) == 1:
        loader = DataLoader(dataset["train"][text_column[0]], batch_size=32, shuffle=False)
        targets_loader = DataLoader(dataset["train"][label_column], batch_size=32, shuffle=False)
    else:
        raise NotImplementedError

    entropy = []
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    for batch, targets in tqdm(zip(loader, targets_loader)):
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
            targets = targets.to(model.device)
        entropy.extend(criterion(outputs.logits, targets).cpu().numpy().tolist())

    with open(f".cache/{dataset}-expected-gradients.json") as f:
        json.dump(expected_gradients, f)

    labels = [(i, l) for i, l in enumerate(dataset['train'][label_column])]
    id2label = dict(enumerate(dataset["train"].features[label_column].names))
    num_examples_per_class = 32 // len(id2label.keys())
    selected_examples = []

    for label_idx in id2label.keys():
        class_entropy = [ent for label, ent in zip(labels, entropy) if label[1] == label_idx]
        class_labels = [label for label in labels if label[1] == label_idx]
        tuples = [(i, l, d) for (i, l), d in zip(class_labels, class_entropy)]
        highest_entropy = sorted(tuples, key=lambda x: x[2], reverse=True)
        selected_examples.extend([idx for idx, _, _ in highest_entropy[:num_examples_per_class]])

    dataset["train"] = dataset["train"].select(selected_examples)
    return dataset
