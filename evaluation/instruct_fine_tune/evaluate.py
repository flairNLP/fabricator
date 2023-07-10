import os
import random

from datasets import load_dataset
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.prompts import TokenLabelPrompt

ner_prompt = (
    "Write a response to the question or task specified in the instruction. "
    "Note the input that provides further context.\n\n"
    "Instruction:\n{instruction}\n\nResponse:"
),

def main():
    dataset = load_dataset("conll2003", split="validation")
    fewshot_examples = dataset.select(random.sample(range(len(dataset)), 3))

    prompt = TokenLabelPrompt(
        input_variables=["tokens"],
        target_variable="ner_tags",
        label_options={0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"},
        task_description="Label the named entities in the sentence.",
    )

    raw_prompt = prompt.get_prompt_text(dataset)
    print(raw_prompt)

    unlabeled_data = dataset

    prompt_node = PromptNode(
        model_name_or_path="EleutherAI/pythia-70M-deduped",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_length=256
    )

    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        support_examples=fewshot_examples,
        unlabeled_examples=unlabeled_data,
        prompt_template=prompt,
        max_prompt_calls=1,
        support_examples_per_prompt=3,
    )


if __name__ == "__main__":
    main()