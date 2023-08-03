import os
from datasets import load_dataset, concatenate_datasets
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator, BasePrompt
from ai_dataset_generator.dataset_transformations.text_classification import convert_label_ids_to_texts


def run():
    dataset = load_dataset("trec", split="train").shuffle(seed=42)
    fewshot_datasets = []
    annotation_datasets = []
    for label in range(6):
        filtered_ds = dataset.filter(lambda x: x["coarse_label"] == label)
        fewshot_datasets.append(filtered_ds.select(range(6)))
        annotation_datasets.append(filtered_ds.select(range(6, len(filtered_ds))))
    fewshot_dataset = concatenate_datasets(fewshot_datasets).shuffle(seed=42)
    annotation_dataset = concatenate_datasets(annotation_datasets).shuffle(seed=42)

    extended_mapping = {
        0: "abbreviation",
        1: "entity",
        2: "description",
        3: "human",
        4: "location",
        5: "number"
    }

    annotation_dataset, label_options = convert_label_ids_to_texts(
        annotation_dataset,
        "coarse_label",
        expanded_label_mapping=extended_mapping,
        return_label_options=True,
    )

    fewshot_dataset = convert_label_ids_to_texts(
        fewshot_dataset,
        "coarse_label",
        expanded_label_mapping=extended_mapping,
    )

    prompt = BasePrompt(
        task_description="Based on the fewshot examples, classify the question into exactly one of the following classes: {}.",
        label_options=label_options,
        generate_data_for_column="coarse_label",
        fewshot_example_columns="text",
        fewshot_formatting_template="Question: {text}\nClass: {coarse_label}",
        target_formatting_template="Question: {text}\nClass: ",
    )

    prompt_node = PromptNode(
        model_name_or_path="gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_length=100,
    )

    generator = DatasetGenerator(prompt_node)
    generated_dataset, original_dataset = generator.generate(
        prompt_template=prompt,
        fewshot_dataset=fewshot_dataset,
        fewshot_examples_per_class=2,
        fewshot_label_sampling_strategy="stratified",
        unlabeled_dataset=annotation_dataset,
        max_prompt_calls=len(annotation_dataset),
        return_unlabeled_dataset=True
    )

    generated_dataset.push_to_hub("trec_annotated_36_fewshot_examples_2_per_prompt_stratified", private=True)
    original_dataset.push_to_hub("trec_original", private=True)


if __name__ == "__main__":
    run()
