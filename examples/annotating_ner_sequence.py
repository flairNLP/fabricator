import random
import os

from datasets import load_dataset
from haystack.nodes import PromptNode

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.task_templates import SequenceLabelDataPoint
from ai_dataset_generator.prompt_templates import NamedEntityAnnotationPrompt



def annotate_ner_data():
    num_support = 10
    num_unlabeled = 100
    total_examples = num_support + num_unlabeled

    dataset = load_dataset("conll2003", split="train")
    dataset = dataset.select(random.sample(range(len(dataset)), total_examples))

    # Get NER tags and apply to datapoints

    tags_key = "pos_tags"
    features = dataset.features
    feature_tags = features[tags_key].feature.names
    ner_tags = {tag: idx for idx, tag in enumerate(list(feature_tags))}
    print(ner_tags)
    ner_samples = [
        SequenceLabelDataPoint(tokens=sample["tokens"], annotations=sample[tags_key], tags=ner_tags)
        for sample in dataset
    ]
    # or just use
    # ner_samples = SequenceLabelDataPoint.build_data_points_from_dataset(dataset, "ner_tags")

    unlabeled_examples, support_examples = ner_samples[:num_unlabeled], ner_samples[num_unlabeled:]

    prompt_template = NamedEntityAnnotationPrompt(tags=ner_tags)
    prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
    generator = DatasetGenerator(prompt_node)
    generated_dataset = generator.generate(
        unlabeled_examples=unlabeled_examples,
        support_examples=support_examples,
        prompt_template=prompt_template,
        max_prompt_calls=1,
    )
    print(generated_dataset)


if __name__ == "__main__":
    annotate_ner_data()
