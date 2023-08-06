from argparse import ArgumentParser
from datasets import load_dataset
from fabricator import convert_spans_to_token_labels
from seqeval.metrics import accuracy_score, f1_score


def run(args):
    id2label = {
        0: "O",
        1: "B-person",
        2: "I-person",
        3: "B-organization",
        4: "I-organization",
        5: "B-location",
        6: "I-location",
        7: "B-miscellaneous",
        8: "I-miscellaneous",
    }
    dataset = load_dataset(args.corpus, split="train")
    dataset = convert_spans_to_token_labels(dataset, "tokens", "ner_tags", id2label=id2label)
    original = load_dataset("conll2003", split="validation")
    y_pred = []
    y_true = []
    for generated_example, original_example in zip(dataset, original):
        if len(generated_example["tokens"]) == len(original_example["tokens"]):
            y_pred.append([id2label[tag] for tag in generated_example["ner_tags"]])
            y_true.append([id2label[tag] for tag in original_example["ner_tags"]])

    print(len(y_pred) / len(dataset))
    print(accuracy_score(y_true, y_pred))
    print(f1_score(y_true, y_pred))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus", type=str)
    arguments = parser.parse_args()
    run(arguments)
