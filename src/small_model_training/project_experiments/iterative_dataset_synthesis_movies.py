from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt

label_options = ["correct", "wrong"]

prompt = BasePrompt(
    task_description="Generate a grammatically {} sentence similar to this:",
    label_options=label_options,
    target_formatting_template="",
)

prompt_node = PromptNode(
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
    max_length=50,
    model_kwargs={
        "model_kwargs": {"do_sample": True, "temperature": 0.5, "top_p": 0.9}
    },
)

generator = DatasetGenerator(prompt_node)
generated_dataset = generator.generate(
    prompt_template=prompt,
    max_prompt_calls=100,
    num_samples_to_generate=100,
    small_model_training="text_classification",
    train_small_model_every_X_generations=10,
)

generated_dataset.push_to_hub("iterative-dataset-synthesis-movies")
