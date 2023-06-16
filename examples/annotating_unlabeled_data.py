import random

from datasets import load_dataset
from langchain.llms import OpenAI

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.task_templates import ExtractiveQADataPoint
from ai_dataset_generator.prompt_templates import AnswerAnnotationPrompt


def annotate_unlabeled_data():
    num_support = 10
    num_unlabeled = 100
    total_examples = num_support + num_unlabeled

    dataset = load_dataset("squad_v2", split="train")
    dataset = dataset.select(random.sample(range(len(dataset)), total_examples))

    extractive_qa_samples = [
        ExtractiveQADataPoint(
            title=sample["title"],
            question=sample["question"],
            context=sample["context"],
            answer=sample["answers"]["text"][0] if sample["answers"]["text"] else None,
            answer_start=sample["answers"]["answer_start"][0] if sample["answers"]["answer_start"] else None,
        ) for sample in dataset]

    unlabeled_examples, support_examples = extractive_qa_samples[:num_unlabeled], extractive_qa_samples[num_unlabeled:]

    prompt_template = AnswerAnnotationPrompt()
    llm = OpenAI(model_name="text-davinci-003")
    generator = DatasetGenerator(llm)
    generated_dataset = generator.generate(
        unlabeled_examples=unlabeled_examples,
        support_examples=support_examples,
        prompt_template=prompt_template,
        max_prompt_calls=1,
    )

if __name__ == "__main__":
    annotate_unlabeled_data()
