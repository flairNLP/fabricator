import os
from datasets import load_dataset, concatenate_datasets
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator, BasePrompt
from fabricator.dataset_transformations.text_classification import convert_label_ids_to_texts


def run():
    annotation_dataset, label_options = convert_label_ids_to_texts(
        load_dataset("snli", split="train").filter(lambda x: x["label"] in [0,1,2]).shuffle(seed=42).select(
            range(10000)),
        "label",
        return_label_options=True,
    )

    fewshot_datasets = []
    for label in range(3):
        filtered_ds = load_dataset("snli", split="validation").filter(lambda x: x["label"] == label)
        fewshot_datasets.append(filtered_ds.select(range(6)))
    fewshot_dataset = concatenate_datasets(fewshot_datasets).shuffle(seed=42)

    fewshot_dataset = convert_label_ids_to_texts(fewshot_dataset, "label")

    prompt = BasePrompt(
        task_description="Given two sentences, determine by means of the fewshot examples whether these sentence "
                         "pairs are: {}.",
        label_options=label_options,
        generate_data_for_column="label",
        fewshot_example_columns=["premise", "hypothesis"],
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
        fewshot_sampling_strategy="stratified",
        unlabeled_dataset=annotation_dataset,
        max_prompt_calls=len(annotation_dataset),
        return_unlabeled_dataset=True
    )

    generated_dataset.push_to_hub("snli_annotated_18_fewshot_examples_2_per_prompt_stratified", private=True)
    original_dataset.push_to_hub("snli_original", private=True)


if __name__ == "__main__":
    run()
