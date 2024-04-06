from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt

label_options = ["positive", "negative"]

prompt = BasePrompt(
    task_description="Generate a {} movie review.",
    label_options=label_options,
)

prompt_node = PromptNode(
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
    max_length=100,
)

generator = DatasetGenerator(prompt_node)
generated_dataset = generator.generate(
    prompt_template=prompt,
    max_prompt_calls=100,
    num_samples_to_generate=100,
    small_model_training='text_classification',
    train_small_model_every_X_generations=10,
)

generated_dataset.push_to_hub("your-first-generated-dataset")