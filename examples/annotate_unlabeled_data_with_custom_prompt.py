import os
import random

from datasets import load_dataset
from haystack.nodes import PromptNode

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.task_templates import ExtractiveQADataPoint
from ai_dataset_generator.prompt_templates import AnnotationPrompt


def annotation_with_custom_prompt():
    num_support = 10
    num_unlabeled = 100
    total_examples = num_support + num_unlabeled

    dataset = load_dataset("squad_v2", split="train")
    dataset = dataset.select(random.sample(range(len(dataset)), 2*total_examples))

    extractive_qa_samples = [
        ExtractiveQADataPoint(
            title=sample["title"],
            question=sample["question"],
            context=sample["context"],
            answer=sample["answers"]["text"][0] if sample["answers"]["text"] else None,
            answer_start=sample["answers"]["answer_start"][0] if sample["answers"]["answer_start"] else None,
        ) for sample in dataset]

    # filter out samples without an answer
    extractive_qa_samples = [sample for sample in extractive_qa_samples if sample.answer is not None][:total_examples]

    unlabeled_examples, support_examples = extractive_qa_samples[:num_unlabeled], extractive_qa_samples[num_unlabeled:]

    custom_annotation_prompt = AnnotationPrompt(
        task_description="Given a question and a answer, generate a text to which the question and answer fits.",
        support_set_variables=["context", "question", "answer"],
        support_set_formatting_template="""Question: {question}\nAnswer: {answer}\nText: {context}""",
        annotation_variables=["question", "answer"],
        annotation_formatting_template="""Question: {question}\nAnswer: {answer}\nText: """
    )

    prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        unlabeled_examples=unlabeled_examples,
        support_examples=support_examples,
        prompt_template=custom_annotation_prompt,
        max_prompt_calls=1,
    )

if __name__ == "__main__":
    annotation_with_custom_prompt()
