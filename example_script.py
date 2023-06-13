import random

from datasets import load_dataset
from langchain.llms import OpenAI

from ai_dataset_generator.task_templates import ExtractiveQADataPoint
from ai_dataset_generator.dataset_generators import DatasetGenerator
from ai_dataset_generator.prompt_templates.question_answering import QuestionGenerationPrompt, AnswerGenerationPrompt


def run():
    # load an arbitrary dataset from huggingface, flair, etc.
    dataset = load_dataset("squad_v2")
    num_unlabeled = 10
    num_support = 10
    total_examples = num_unlabeled + num_support

    # this is still custom work. Iterate over your dataset and create a list of TaskDataPoints.
    # these classes are TaskSpecific, for instance for QA, NER or text classification
    # they ensure uniform data structures for the dataset generation
    extractive_qa_samples = []
    while len(extractive_qa_samples) < total_examples:
        i = random.randint(0, len(dataset["train"]))
        sample = dataset["train"][i]
        if not sample["answers"]["text"]:
            continue
        extractive_qa_samples.append(ExtractiveQADataPoint(
            title=sample["title"],
            question=sample["question"],
            context=sample["context"],
            answer=sample["answers"]["text"][0],
            answer_start=sample["answers"]["answer_start"][0] if sample["answers"]["answer_start"] else None,
        ))

    support_examples = extractive_qa_samples[:num_support]
    unlabeled_examples = extractive_qa_samples[num_support:]

    # nice prompting interface using langchain
    # task-specific generator, list of task-specific data points are passed as well as a prompt node
    llm = OpenAI(model_name="text-davinci-003")
    #prompt_template = QuestionGenerationPrompt()
    prompt_template = AnswerGenerationPrompt()
    generator = DatasetGenerator(llm)
    generated_dataset = generator.generate(
        unlabeled_examples=unlabeled_examples,
        support_examples=support_examples,
        prompt_template=prompt_template,
    )

if __name__ == "__main__":
    run()
