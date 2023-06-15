import random
import sys
sys.path.append("..")

from datasets import load_dataset
from langchain.llms import OpenAI
from dotenv import load_dotenv

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.task_templates import SequenceLabelDataPoint
from ai_dataset_generator.prompt_templates import NamedEntityAnnotationPrompt

load_dotenv()

def annotate_ner_data():
    num_support = 10
    num_unlabeled = 100
    total_examples = num_support + num_unlabeled

    dataset = load_dataset("conll2003", split="train")
    dataset = dataset.select(random.sample(range(len(dataset)), total_examples))
    
    # This can be switched out for Chunking or POS
    ner_samples = [SequenceLabelDataPoint(tokens=sample["tokens"], annotations=sample["ner_tags"]) for sample in dataset]

    unlabeled_examples, support_examples = ner_samples[:num_unlabeled], ner_samples[num_unlabeled:]

    prompt_template = NamedEntityAnnotationPrompt()
    llm = OpenAI(model_name="text-davinci-003")
    generator = DatasetGenerator(llm)
    generated_dataset = generator.generate(
        unlabeled_examples=unlabeled_examples,
        support_examples=support_examples,
        prompt_template=prompt_template,
        max_prompt_calls=1,
    )
    print(generated_dataset)

if __name__ == "__main__":
    annotate_ner_data()
