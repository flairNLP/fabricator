import os
import random

from datasets import load_dataset
from haystack.nodes import PromptNode

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.prompt_templates import TextGenerationPrompt
from ai_dataset_generator.task_templates import TextDataPoint


def run_unlabeled_generation():
    num_samples = 10

    dataset = load_dataset("trec", split="train")
    dataset = dataset.select(random.sample(range(len(dataset)), num_samples))

    support_examples = [TextDataPoint(text=sample["text"]) for sample in dataset]

    generation_prompt = TextGenerationPrompt()
    prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        support_examples=support_examples,
        prompt_template=generation_prompt,
        max_prompt_calls=1,
    )


if __name__ == "__main__":
    run_unlabeled_generation()
