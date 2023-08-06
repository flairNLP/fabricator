import argparse
import os
import random

from datasets import load_dataset, load_from_disk
from haystack.nodes import PromptNode

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.prompts import BasePrompt
from ai_dataset_generator.samplers import random_sampler
from ai_dataset_generator.dataset_transformations.text_classification import convert_label_ids_to_texts


ner_prompt = (
    "Given the following text. Annotate the example and choose your annotations from: {}"
)

def main(args):

    # model_name_or_path = "EleutherAI/pythia-70M-deduped"
    dataset = load_dataset(args.dataset_name, split=args.split)
    fewshot_examples = dataset.select(random.sample(range(len(dataset)), 3))


    prompt = BasePrompt(
        task_description=ner_prompt,
        generate_data_for_column="ner_tags",
        fewshot_example_columns="tokens",
        label_options=None #{"O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"},
    )

    unlabeled_data = random_sampler(dataset, 30)

    if not args.use_cached:

        # "tiiuae/falcon-7b-instruct"
        # timdettmers/guanaco-33b-merged
        prompt_node = PromptNode(
            model_name_or_path=args.model_name_or_path,
            api_key=os.environ.get("HF_API_KEY"),
        )


        generator = DatasetGenerator(prompt_node)
        generated_dataset = generator.generate(
            prompt_template=prompt,
            fewshot_dataset=None,
            unlabeled_dataset=unlabeled_data,
            max_prompt_calls=30,
            fewshot_examples_per_class=0,
            timeout_per_prompt=2,
        )

        generated_dataset.save_to_disk("generated_dataset_starchat")

    else:
        generated_dataset = load_from_disk("generated_dataset")


    evaluate(dataset, generated_dataset)


def post_process(generated_samples):
    """Some heuristics to clean up the generated samples"""

    def _post_process(generated_sample):

        cleaned_tags = []

        for tag in generated_sample["ner_tags"]:
            try:
                cleaned_tags.append(int(tag))
            except ValueError:
                if tag == "-":
                    cleaned_tags.append(0)
                elif tag.startswith("[") and tag.endswith("]") and len(tag) > 2:
                    try:
                        cleaned_tags.append(int(tag[1:-1]))
                    except ValueError:
                        cleaned_tags.append(0)

        generated_sample["ner_tags"] = cleaned_tags

        return generated_sample

    return generated_samples.map(_post_process)

def build_gold_and_prediction_pairs(dataset, generated_dataset):
    """Builds a list of gold and predicted labels for each sample in the dataset"""

    golds = []
    predictions = []

    for generated_sample in generated_dataset:

        for gold_sample in dataset:

            if generated_sample["tokens"] == gold_sample["tokens"]:
                golds.append(gold_sample["ner_tags"])
                predictions.append(generated_sample["ner_tags"])


    return golds, predictions

def calculate_metrics(golds, predictions):
    mlb = MultiLabelBinarizer()
    golds = mlb.fit_transform(golds)
    predictions = mlb.transform(predictions)
    acc = accuracy_score(golds, predictions)
    report = classification_report(golds, predictions)
    # Print the results
    print(f"Accuracy: {acc}")
    print(f"Classification Report:\n{report}")


def evaluate(dataset, generated_dataset):
    generated_dataset = post_process(generated_dataset)
    print(f"Using {generated_dataset} of samples from the generated dataset")
    golds, predictions = build_gold_and_prediction_pairs(dataset, generated_dataset)
    calculate_metrics(golds, predictions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/pythia-70M-deduped")
    parser.add_argument("--dataset_name", type=str, default="conll2003")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--use_cached", type=bool, default=False)

    args = parser.parse_args()

    main(args)
