import os
from datasets import load_dataset
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.prompt_templates import RefactoredBasePrompt

"""
For Demo:
1) How to create prompts for
1.a) unlabeled
1.b) if you want to annotate some variable

2) How to annotate unlabeled data (trec, imdb, glue, squad, conll2003)
2.a) Show task suffix when using classification + using label2idx.
2.b) Show easy switch between task layout with glue.
2.c) Show shift of preprocessing now done by user at example of squad and conll.

3) Adjusted Generation Function - returns now Dataset object
3.a) Show we can push to hub now
3.b) Postprocessing after generation. Show demo for NER + QA

"""

def prompt_for_unlabeled():
    dataset = load_dataset("imdb", split="train")
    support_set = dataset.select([1, 2, 3])

    input_variables = ["text"]
    output_format = "text"
    prompt = RefactoredBasePrompt(
        input_variables=input_variables,
        output_format=output_format,
        task_description="Generate similar texts.",
    )
    raw_prompt = prompt.get_prompt_text(support_set)
    print(raw_prompt)

def prompt_for_annotation():
    # Load some datasets
    dataset = load_dataset("imdb", split="train")
    support_set = dataset.select([50, 55, 60, 65])

    # Column names as they occur in the dataset
    input_variables = ["text"]  # Inputs can be many texts, so either be a List[str] or a str
    target_variable = "label"  # Target / annotation variable needs to be a str
    output_format = "single_label_classification" # Annotation format can be "text", "single_label", "multi_label", "token_classification" and determines how the LLM is prompted for the annotation
    idx2label = {idx: key for idx, key in enumerate(support_set.features[target_variable].feature.names)}

    prompt = RefactoredBasePrompt(
        input_variables=input_variables,
        output_format=output_format,
        target_variable=target_variable,
        classification_options=idx2label,
        task_description="Annotate each token with exactly one label."
    )
    raw_prompt = prompt.get_prompt_text(support_set)
    print(raw_prompt)

def generate_unlabeled():
    dataset = load_dataset("imdb", split="train")
    support_set = dataset.select([1, 2, 3])
    input_variables = ["text"]
    output_format = "text"
    prompt = RefactoredBasePrompt(
        input_variables=input_variables,
        output_format=output_format,
        task_description="Generate similar texts.",
    )
    raw_prompt = prompt.get_prompt_text(support_set)
    print(raw_prompt)

    prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"), max_length=100)
    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        support_examples=support_set,
        prompt_template=prompt,
        max_prompt_calls=3,
        support_examples_per_prompt=1,
    )
    print()

def generate_annotations():
    # Load some datasets
    dataset = load_dataset("imdb", split="train")
    to_annotate = dataset.select([100, 105, 110])
    support_set = dataset.select([50, 55, 60, 65])

    # Column names as they occur in the dataset
    input_variables = ["text"]  # Inputs can be many texts, so either be a List[str] or a str
    target_variable = "label"  # Target / annotation variable needs to be a str
    output_format = "single_label_classification" # Annotation format can be "text", "single_label", "multi_label", "token_classification" and determines how the LLM is prompted for the annotation
    idx2label = {idx: key for idx, key in enumerate(support_set.features[target_variable].feature.names)}

    prompt = RefactoredBasePrompt(
        input_variables=input_variables,
        output_format=output_format,
        target_variable=target_variable,
        classification_options=idx2label,
        task_description="Classify the movie review."
    )
    raw_prompt = prompt.get_prompt_text(support_set)
    print(raw_prompt)

    prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
    generator = DatasetGenerator(prompt_node)
    generated_dataset, original_dataset = generator.generate(
        support_examples=support_set,
        unlabeled_examples=to_annotate,
        prompt_template=prompt,
        max_prompt_calls=3,
        support_examples_per_prompt=1,
        return_original_dataset=True,
    )
    print()


if __name__ == "__main__":
    #prompt_for_unlabeled()
    #prompt_for_annotation()
    #generate_unlabeled()
    generate_annotations()

"""
Support functions:

for example in original_dataset:
    print(f"{example['context']}\n{example['question']}\n{example['answer']}\n")

for example in original_dataset:
    print(f"{example['tokens']}\n{example['ner_tags']}\n")

# Squad Samples with answer: 100, 105, 110, 50, 55, 60, 65
def preprocess(example):
    if example["answer"]:
        example["answer"] = example["answer"].pop()
    else:
        example["answer"] = ""
    return example

# Push to hub
generated_dataset.push_to_hub("whoisjones/test-generation", private=True)
original_dataset.push_to_hub("whoisjones/test-original", private=True)

# Postprocess NER
def postprocess(example):
    import re
    processed_labels = []
    for label in example["ner_tags"]:
        new_label = re.findall(r'\d+', label)
        if new_label:
            processed_labels.append(int(new_label.pop()))
        else:
            processed_labels.append(0)
    example["ner_tags"] = processed_labels
    return example

# Postprocess QA
def postprocess(example):
    answer_start = example["context"].find(example["answer"])
    if answer_start < 0:
        print(
            f'Could not calculate the answer start because the context "{example["context"]}" '
            f'does not contain the answer "{example["answer"]}".'
        )
        answer_start = -1
    else:
        # check that the answer doesn't occur more than once in the context
        second_answer_start = example["context"].find(example["answer"], answer_start + 1)
        if second_answer_start >= 0:
            print(
                "Could not calculate the answer start because the context contains the answer more than once."
            )
            answer_start = -1
        else:
            answer_start = answer_start
    example["answer_start"] = answer_start
    return example

"""