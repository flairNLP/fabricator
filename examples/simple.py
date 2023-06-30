import os
import sys
sys.path.append(".")

from datasets import load_dataset
from ai_dataset_generator.prompts import DataGenerationPrompt
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator

def main():
    dataset = load_dataset("imdb", split="train")
    fewshot_examples = dataset.select([1, 2, 3])

    input_variables = ["text"] # Column names as they occur in the dataset
    output_format = "text" # indicates the output format of the LLM is text
    prompt = DataGenerationPrompt(
        input_variables=input_variables,
        output_format=output_format,
        task_description="Generate similar texts.",
    )
    raw_prompt = prompt.get_prompt_text(fewshot_examples)

# os.environ.get("OPENAI_API_KEY")
    # prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key="", max_length=100)
    generator = DatasetGenerator(None)
    generated_dataset = generator.generate(
        support_examples=fewshot_examples, # from above
        prompt_template=prompt, # from above
        max_prompt_calls=3, # max number of calls to the LLM
        support_examples_per_prompt=1, # number of support examples per prompt
        dry_run=True
    )

    # generated_dataset.push_to_hub("your-first-generated-dataset")
    breakpoint()
    print(raw_prompt)


if __name__ == '__main__':
    main()
