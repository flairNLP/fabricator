import os
from datasets import load_dataset, concatenate_datasets
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator, BasePrompt
from fabricator.dataset_transformations.text_classification import convert_label_ids_to_texts

def run():
    for possible_examples_per_class, fewshot_example_per_class in [(0,0), (2,1), (2,2), (4,1), (4,2), (4,3), (4,4), (8,1), (8,2), (8,3),
                                                                   (8,4), (16,1), (16,2), (16,3), (16,4)]:
        dataset = load_dataset("trec", split="train").shuffle(seed=42).train_test_split(500, stratify_by_column="coarse_label")
        fewshot_dataset = dataset["train"]
        annotation_dataset = dataset["test"]
        fewshot_datasets = []
        for label in range(6):
            filtered_ds = fewshot_dataset.filter(lambda x: x["coarse_label"] == label)
            fewshot_datasets.append(filtered_ds.select(range(possible_examples_per_class)))
        fewshot_dataset = concatenate_datasets(fewshot_datasets).shuffle(seed=42)

        extended_mapping = {
            0: "abbreviation",
            1: "entity",
            2: "description",
            3: "human",
            4: "location",
            5: "number"
        }

        if possible_examples_per_class > 0:
            fewshot_dataset = convert_label_ids_to_texts(
                fewshot_dataset,
                "coarse_label",
                expanded_label_mapping=extended_mapping,
            )

        annotation_dataset, label_options = convert_label_ids_to_texts(
            annotation_dataset,
            "coarse_label",
            expanded_label_mapping=extended_mapping,
            return_label_options=True,
        )

        prompt = BasePrompt(
            task_description="Classify the question into exactly one of the following classes: {}.",
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
        generated_dataset = generator.generate(
            prompt_template=prompt,
            fewshot_dataset=fewshot_dataset if possible_examples_per_class > 0 else None,
            fewshot_examples_per_class=fewshot_example_per_class if possible_examples_per_class > 0 else 0,
            fewshot_sampling_strategy="stratified" if possible_examples_per_class > 0 else None,
            fewshot_sampling_column="coarse_label" if possible_examples_per_class > 0 else None,
            unlabeled_dataset=annotation_dataset,
            max_prompt_calls=len(annotation_dataset),
        )

        model_name = f"trec_hyperparameter_annotated_{possible_examples_per_class}_possible_examples_{fewshot_example_per_class}_used"
        generated_dataset.push_to_hub(model_name, private=True)


if __name__ == "__main__":
    run()
