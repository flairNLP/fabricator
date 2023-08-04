import os
from datasets import load_dataset, concatenate_datasets
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator, BasePrompt
from ai_dataset_generator.dataset_transformations.text_classification import convert_label_ids_to_texts

def run():
    for possible_examples_per_class, fewshot_example_per_class in [(0,0), (2,2), (4,2), (4,3), (4,4), (8,2), (8,3),
                                                                   (8,4), (8,5), (16,2), (16,3), (16,4), (16,5)]:
        dataset = load_dataset("trec", split="train").shuffle(seed=42)
        if possible_examples_per_class > 0:
            fewshot_datasets = []
            for label in range(6):
                filtered_ds = dataset.filter(lambda x: x["coarse_label"] == label)
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

            fewshot_dataset, label_options = convert_label_ids_to_texts(
                fewshot_dataset,
                "coarse_label",
                expanded_label_mapping=extended_mapping,
                return_label_options=True,
            )
            task_description = "Generate a new question that asks about: {}. The new question should be very " \
                               "different from the fewshot examples."
        else:
            task_description = "Generate a new question that asks about: {}."
            label_options = ["abbreviation", "entity", "description", "human", "location", "number"]
            fewshot_dataset = None

        prompt = BasePrompt(
            task_description=task_description,
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
            fewshot_examples_per_class=fewshot_example_per_class,
            fewshot_sampling_strategy="uniform",
            fewshot_sampling_column="coarse_label",
            max_prompt_calls=500,
            num_samples_to_generate=500,
        )

        model_name = f"trec_generated_{possible_examples_per_class}_possible_examples_{fewshot_example_per_class}_used"
        generated_dataset.push_to_hub(model_name, private=True)


if __name__ == "__main__":
    run()
