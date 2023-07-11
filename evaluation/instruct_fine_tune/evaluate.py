import os
import random

from datasets import load_dataset, load_from_disk
from haystack.nodes import PromptNode

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.prompts import TokenLabelPrompt


ner_prompt = (
    "Write a response to the question or task specified in the instruction. "
    "Note the input that provides further context.\n\n"
    "Instruction:\n{instruction}\n\nResponse:"
)

ner_prompt = (
    "Given the following text. Annotate the example and choose your annotations from: {label_options}"
)

def main():

    use_cached = True

    model_name_or_path = "EleutherAI/pythia-70M-deduped"
    dataset = load_dataset("conll2003", split="validation")
    fewshot_examples = dataset.select(random.sample(range(len(dataset)), 3))

    prompt = TokenLabelPrompt(
        input_variables=["tokens"],
        target_variable="ner_tags",
        label_options={0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"},
        task_description=ner_prompt
    )

    raw_prompt = prompt.get_prompt_text(dataset)
    # print(raw_prompt)

    unlabeled_data = dataset

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # prompt_node = PromptNode(
    #     model_name_or_path,
    #     model_kwargs={"model": model, "tokenizer": tokenizer}
    # )

    if not use_cached:

        prompt_node = PromptNode(
            model_name_or_path="text-davinci-003",
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_length=100
        )

        generator = DatasetGenerator(prompt_node)
        generated_dataset = generator.generate(
            support_examples=fewshot_examples,
            unlabeled_examples=unlabeled_data,
            prompt_template=prompt,
            max_prompt_calls=3,
            support_examples_per_prompt=3,
        )

        generated_dataset.save_to_disk("generated_dataset")

    else:
        generated_dataset = load_from_disk("generated_dataset")


    evaluate(dataset, generated_dataset)


def post_process(generated_samples):

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
    golds, predictions = build_gold_and_prediction_pairs(dataset, generated_dataset)
    calculate_metrics(golds, predictions)


if __name__ == "__main__":
    main()