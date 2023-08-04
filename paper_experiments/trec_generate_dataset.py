import os
from datasets import load_dataset, concatenate_datasets
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator, BasePrompt
from fabricator.dataset_transformations.text_classification import convert_label_ids_to_texts


def run():
    dataset = load_dataset("trec", split="train").shuffle(seed=42)
    fewshot_datasets = []
    for label in range(6):
        filtered_ds = dataset.filter(lambda x: x["coarse_label"] == label)
        fewshot_datasets.append(filtered_ds.select(range(8)))
    fewshot_dataset = concatenate_datasets(fewshot_datasets).shuffle(seed=42)

    extended_mapping = {
        0: "abbreviation",
        1: "entity",
        2: "description",
        3: "human",
        4: "location",
        5: "number"
    }

    fewshot_dataset, label_options = convert_label_ids_to_texts(
        fewshot_dataset,
        "coarse_label",
        expanded_label_mapping=extended_mapping,
        return_label_options=True,
    )

    prompt = BasePrompt(
        task_description="Generate a new question that asks about: {}. The new question should be very different from "
                         "the fewshot examples.",
        label_options=label_options,
        generate_data_for_column="text",
        fewshot_formatting_template="Question: {text}",
        target_formatting_template="Question: ",
    )

    prompt_node = PromptNode(
        model_name_or_path="gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_length=100,
    )

    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        prompt_template=prompt,
        fewshot_dataset=fewshot_dataset,
        fewshot_examples_per_class=3,
        fewshot_label_sampling_strategy="uniform",
        fewshot_sampling_column="coarse_label",
        max_prompt_calls=10000,
        num_samples_to_generate=10000,
    )

    generated_dataset.push_to_hub("trec_generate_48_fewshot_examples_3_per_prompt_stratified", private=True)


if __name__ == "__main__":
    run()
