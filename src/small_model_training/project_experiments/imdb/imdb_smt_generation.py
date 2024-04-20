from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt

label_options = ["positive", "negative"]

prompt = BasePrompt(
    task_description="Generate a {} movie review similar to this:",
    label_options=label_options,
    target_formatting_template="",
)

prompt_node = PromptNode(
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
    max_length=300,
    model_kwargs={"model_kwargs": {"do_sample": True, "temperature": 0.4, "top_p": 1}},

)

generator = DatasetGenerator(prompt_node)
generated_dataset = generator.generate(
    prompt_template=prompt,
    max_prompt_calls=500,
    num_samples_to_generate=500,
    small_model_training='text_classification',
    train_small_model_every_X_generations=50,
)

generated_dataset.push_to_hub("500-movie-reviews-smt")