import random

from ai_dataset_generator.prompt_templates.single_label_classification import \
    AddSingleLabelAnnotationPrompt
from ai_dataset_generator.task_templates.single_label_classification import \
    SingleLabelClassificationDataPoint
from datasets import load_dataset
from langchain.llms import OpenAI
from loguru import logger

from ai_dataset_generator import DatasetGenerator


def annotate_unlabeled_data():
    num_support = 10
    num_unlabeled = 100
    total_examples = num_support + num_unlabeled

    dataset = load_dataset("imdb", split="train")
    dataset = dataset.select(random.sample(range(len(dataset)), total_examples))
    all_labels = list(set(dataset['label']))

    sl_classification_samples = [
        SingleLabelClassificationDataPoint(
            text=sample["text"],
            label=sample["label"],
        )
        for sample in dataset
    ]

    unlabeled_examples, support_examples = (
        sl_classification_samples[:num_unlabeled],
        sl_classification_samples[num_unlabeled:],
    )

    prompt_template = AddSingleLabelAnnotationPrompt(labels=all_labels)
    llm = OpenAI(model_name="text-davinci-003")
    generator = DatasetGenerator(llm)
    generated_dataset = generator.generate(
        unlabeled_examples=unlabeled_examples,
        support_examples=support_examples,
        prompt_template=prompt_template,
        max_prompt_calls=1,
    )
    logger.info(generated_dataset)


if __name__ == "__main__":
    annotate_unlabeled_data()
