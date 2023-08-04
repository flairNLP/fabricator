import os
from datasets import load_dataset, concatenate_datasets
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator, BasePrompt
from ai_dataset_generator.dataset_transformations.text_classification import convert_label_ids_to_texts


def run():
    annotation_dataset, label_options = convert_label_ids_to_texts(
        load_dataset("glue", "mrpc", split="train"),
        "label",
        return_label_options=True,
    )
    fewshot_datasets = []
    for label in range(2):
        filtered_ds = load_dataset("glue", "mrpc", split="validation").filter(
            lambda x: x["label"] == label)
        fewshot_datasets.append(filtered_ds.select(range(6)))
    fewshot_dataset = concatenate_datasets(fewshot_datasets).shuffle(seed=42)

    fewshot_dataset = convert_label_ids_to_texts(fewshot_dataset, "label")

    prompt = BasePrompt(
        task_description="Given two sentences, determine by means of the fewshot examples whether these sentences are: {}.",
        label_options=label_options,
        generate_data_for_column="label",
        fewshot_example_columns=["sentence1", "sentence2"],
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

    generated_dataset.push_to_hub("glue_mrpc_annotated_12_fewshot_examples_2_per_prompt_stratified", private=True)
    original_dataset.push_to_hub("glue_mrpc_original", private=True)


if __name__ == "__main__":
    run()
