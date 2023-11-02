import json
from argparse import Namespace
from collections import Counter
from tqdm import tqdm

import numpy as np

from utils import *


def select_fewshots(
    args: Namespace,
    full_dataset: DatasetDict,
    dataset_size: int,
    task_keys: dict,
) -> DatasetDict:
    """
    Selects a fewshot dataset from the full dataset according to the specified init_strategy.
    """

    if args.init_strategy == "random":
        dataset = random_selection(
            full_dataset,
            dataset_size,
            task_keys["label_column"]
        )
    elif args.init_strategy == "class-centeroid-closest":
        if args.embedding_model is None:
            raise ValueError("You need to specify an embedding model for this init strategy.")
        dataset = closest_to_centeroid_selection(
            args.embedding_model,
            full_dataset,
            dataset_size,
            task_keys
        )
    elif args.init_strategy == "class-centeroid-furthest":
        if args.embedding_model is None:
            raise ValueError("You need to specify an embedding model for this init strategy.")
        dataset = furthest_to_centeroid_selection(
            args.embedding_model,
            full_dataset,
            dataset_size,
            task_keys
        )
    elif args.init_strategy == "expected-gradients":
        dataset = expected_gradients_selection(
            args.tam_model,
            full_dataset,
            dataset_size,
            task_keys
        )
    elif args.init_strategy == "certainty":
        dataset = entropy_selection(
            args.tam_model,
            full_dataset,
            dataset_size,
            task_keys
        )
    else:
        raise NotImplementedError

    return dataset


def random_selection(
    dataset: DatasetDict,
    num_total_samples: int,
    label_column: str
) -> DatasetDict:
    """
    Selects a fewshot dataset from the full dataset by randomly selecting examples.
    """
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
    model_name_or_path: str,
    dataset: DatasetDict,
    num_total_samples: int,
    task_keys: dict,
) -> DatasetDict:
    """
    Selects a fewshot dataset from the full dataset by selecting the examples closest to the class centeroid.
    """
    return selection_with_class_centeroids(
        model_name_or_path,
        dataset,
        num_total_samples,
        task_keys,
        selection_strategy="closest"
    )

def furthest_to_centeroid_selection(
    model_name_or_path: str,
    dataset: DatasetDict,
    num_total_samples: int,
    task_keys: dict,
) -> DatasetDict:
    """
    Selects a fewshot dataset from the full dataset by selecting the examples furthest to the class centeroid.
    """
    return selection_with_class_centeroids(
        model_name_or_path,
        dataset,
        num_total_samples,
        task_keys,
        selection_strategy="furthest"
    )


def selection_with_class_centeroids(
    model_name_or_path: str,
    dataset: DatasetDict,
    num_total_samples: int,
    task_keys: dict,
    selection_strategy: str,
) -> DatasetDict:
    """
    Selects a fewshot dataset from the full dataset by selecting the examples relative to the class centeroid.
    """
    label_column = task_keys["label_column"]
    id2label = dict(enumerate(dataset["train"].features[label_column].names))

    cache_file_name = f"{dataset}-centeroid-{selection_strategy}.json"

    if cache_file_name in os.listdir(CACHE_DIR):
        with open(os.path.join(CACHE_DIR, cache_file_name), "r") as f:
            distance_to_center = json.load(f)
    else:
        model, tokenizer = get_embedding_model_and_tokenizer(model_name_or_path)
        train_loader = get_trainloader(dataset, tokenizer, task_keys)

        embeddings = []
        with torch.no_grad():
            for batch in tqdm(train_loader):
                outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
                cls_rep = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                embeddings.extend(cls_rep)

        assert len(embeddings) == len(dataset["train"])

        embedding_tuples = [(i, l, e) for i, (l, e) in enumerate(zip(dataset["train"][label_column], embeddings))]

        def cosine(x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        distance_to_center = []
        for label_idx in id2label.keys():
            class_embeddings = [emb for i, label, emb in embedding_tuples if label == label_idx]
            class_centeroid = np.mean(class_embeddings, axis=0)
            dists = [cosine(class_centeroid, emb) for emb in class_embeddings]
            distance_to_center.extend([(i, l, d) for (i, l, e), d in zip(embedding_tuples, dists)])

        with open(os.path.join(CACHE_DIR, cache_file_name), "w") as f:
            json.dump(distance_to_center, f)

    num_examples_per_class = num_total_samples // len(id2label.keys())
    selected_examples = []

    for label_idx in id2label.keys():
        class_distance_to_center = [dist_tuple for dist_tuple in distance_to_center if dist_tuple[1] == label_idx]
        sorted_di = sorted(class_distance_to_center, key=lambda x: x[2], reverse=True if selection_strategy == "closest" else False)
        selected_examples.extend([idx for idx, _, _ in sorted_di[:num_examples_per_class]])

    dataset["train"] = dataset["train"].select(selected_examples)
    return dataset


def expected_gradients_selection(
    model_name_or_path: str,
    dataset: DatasetDict,
    num_total_samples: int,
    task_keys: dict,
) -> DatasetDict:
    label_column = task_keys["label_column"]
    id2label = dict(enumerate(dataset["train"].features[label_column].names))

    cache_file_name = f"{dataset}-expected-gradients.json"

    if cache_file_name in os.listdir(CACHE_DIR):
        with open(os.path.join(CACHE_DIR, cache_file_name), "r") as f:
            expected_gradients = json.load(f)
    else:
        model, tokenizer = get_classification_model_and_tokenizer(model_name_or_path)
        train_loader = get_trainloader(dataset, tokenizer, task_keys)

        expected_gradients = []
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        model.train()
        params_filter = [n for n, p in model.named_parameters() if not p.requires_grad]

        example_id = 0
        with torch.enable_grad():
            for batch in tqdm(train_loader):

                outputs = model(**batch)
                targets = torch.tensor(dataset["train"][label_column][idx * batch_size: (idx + 1) * batch_size]).to(model.device)
                loss = criterion(outputs.logits, targets)

                for single_loss, single_target in zip(loss, targets):
                    gradients = torch.autograd.grad(
                        outputs=single_loss,
                        inputs=[
                            param for name, param
                            in model.named_parameters()
                            if name not in params_filter
                        ],
                        create_graph=True
                    )

                    gradients = [a.detach() for a in gradients]

                    expected_gradients.append((
                        example_id,
                        single_target.cpu().numpy().tolist(),
                        torch.norm(
                            torch.cat(
                                [layer_grad.view(-1) for layer_grad in gradients]
                            ), p=2).cpu().numpy().tolist()
                    ))

                    example_id += 1

        assert len(expected_gradients) == len(dataset["train"])

        with open(os.path.join(CACHE_DIR, cache_file_name, "w")) as f:
            json.dump(expected_gradients, f)

    num_examples_per_class = num_total_samples // len(id2label.keys())
    selected_examples = []

    for i, label in id2label.items():
        class_expected_gradients = [grad_tuple for grad_tuple in expected_gradients if grad_tuple[1] == i]
        sorted_class_expected_gradients = sorted(class_expected_gradients, key=lambda x: x[2], reverse=True)
        selected_examples.extend([grad_tuple[0] for grad_tuple in sorted_class_expected_gradients[:num_examples_per_class]])

    dataset["train"] = dataset["train"].select(selected_examples)
    return dataset


def cross_entropy_selection(
    model_name_or_path: str,
    dataset: DatasetDict,
    num_total_samples: int,
    task_keys: dict,
) -> DatasetDict:
    label_column = task_keys["label_column"]
    id2label = dict(enumerate(dataset["train"].features[label_column].names))

    cache_file_name = f"{dataset}-cross-entropy.json"

    if cache_file_name in os.listdir(CACHE_DIR):
        with open(os.path.join(CACHE_DIR, cache_file_name), "r") as f:
            entropy_tuples = json.load(f)
    else:
        model, tokenizer = get_classification_model_and_tokenizer(model_name_or_path)
        train_loader = get_trainloader(dataset, tokenizer, task_keys)

        entropy = []
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for batch, targets in tqdm(train_loader):
                outputs = model(**batch)
                targets = batch.pop("labels")
                entropy.extend(criterion(outputs.logits, targets).cpu().numpy().tolist())

        assert len(entropy) == len(dataset["train"])

        entropy_tuples = [(i, l, e) for i, (l, e) in enumerate(zip(dataset["train"][label_column], entropy))]

        with open(os.path.join(CACHE_DIR, cache_file_name, "w")) as f:
            json.dump(entropy_tuples, f)

    num_examples_per_class = num_total_samples // len(id2label.keys())
    selected_examples = []

    for label_idx in id2label.keys():
        class_entropy = [ent_tuple for ent_tuple in entropy_tuples if ent_tuple[1] == label_idx]
        sorted_ent = sorted(class_entropy, key=lambda x: x[2], reverse=True)
        selected_examples.extend([idx for idx, _, _ in sorted_ent[:num_examples_per_class]])

    dataset["train"] = dataset["train"].select(selected_examples)
    return dataset

def entropy_selection(
    model_name_or_path: str,
    dataset: DatasetDict,
    num_total_samples: int,
    task_keys: dict,
) -> DatasetDict:
    label_column = task_keys["label_column"]
    id2label = dict(enumerate(dataset["train"].features[label_column].names))

    cache_file_name = f"{dataset}-entropy.json"

    if cache_file_name in os.listdir(CACHE_DIR):
        with open(os.path.join(CACHE_DIR, cache_file_name), "r") as f:
            entropy_tuples = json.load(f)
    else:
        model, tokenizer = get_classification_model_and_tokenizer(model_name_or_path)
        train_loader = get_trainloader(dataset, tokenizer, task_keys)

        entropy = []

        with torch.no_grad():
            for batch, targets in tqdm(train_loader):
                outputs = model(**batch)
                dist = torch.nn.functional.softmax(outputs.logits, dim=-1)
                entropy.extend(torch.distributions.Categorical(dist).entropy().cpu().numpy().tolist())

        assert len(entropy) == len(dataset["train"])

        entropy_tuples = [(i, l, e) for i, (l, e) in enumerate(zip(dataset["train"][label_column], entropy))]

        with open(os.path.join(CACHE_DIR, cache_file_name, "w")) as f:
            json.dump(entropy_tuples, f)

    num_examples_per_class = num_total_samples // len(id2label.keys())
    selected_examples = []

    for label_idx in id2label.keys():
        class_entropy = [ent_tuple for ent_tuple in entropy_tuples if ent_tuple[1] == label_idx]
        sorted_ent = sorted(class_entropy, key=lambda x: x[2], reverse=True)
        selected_examples.extend([idx for idx, _, _ in sorted_ent[:num_examples_per_class]])

    dataset["train"] = dataset["train"].select(selected_examples)
    return dataset
