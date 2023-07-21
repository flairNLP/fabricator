import os
import random
from argparse import ArgumentParser

from datasets import Sequence, Value, load_dataset, concatenate_datasets
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.dataset_transformations.question_answering import (
    preprocess_squad_format,
    postprocess_squad_format,
)
from ai_dataset_generator.prompts import TextLabelPrompt


def run(arguments):
    """Generate answers based on a few-shot example from context and question."""
    org = load_dataset(arguments.dataset, split=arguments.split)


    dataset_name = f"qa-dataset"
    dataset_answerable_questions = org.filter(lambda sample: sample['answers']['text']).shuffle()
    dataset_unanswerable_questions = org.filter(lambda sample: not sample['answers']['text']).shuffle()

    prompt_node = PromptNode(
        model_name_or_path=arguments.llm, api_key=os.environ.get("OPENAI_API_KEY"), max_length=100
    )
    generator = DatasetGenerator(prompt_node)

    question_words = ["Which", "What", "How", "When", "Who", "How many", "Where", "Why"]

    filtered_generated_datasets = []
    filtered_original_datasets = []

    def merge_columns(example):
        if example["answer"] == "":
            example["qa"] = example["question"]
            return example
        example["qa"] = f"{example['question']}\n{example['answer']}"
        return example

    def split_columns(example):
        entries = example["qa"].split("\n")
        example["question"] = entries[0]
        if len(entries) == 1:
            example["answer"] = ""
            return example
        example["answer"] = entries[1]
        return example

    for index, dataset in enumerate([dataset_answerable_questions, dataset_unanswerable_questions]):
        preprocessed_dataset = preprocess_squad_format(dataset)
        preprocessed_dataset = preprocessed_dataset.map(merge_columns)
        fewshot_examples = preprocessed_dataset.select(range(10))
        labels_to_generate = arguments.num_labels//3*2 if index == 0 else arguments.num_labels//3
        unlabeled_examples = preprocessed_dataset.select(range(10, labels_to_generate + 10, 1))


        for i in range(0, len(unlabeled_examples), arguments.save_steps):
            fewshot_examples = fewshot_examples.shuffle().select(range(3))

            task_descriptions = [
                "Given a text, first create a difficult question that can be answered using the text. The question must describe the context of the text. Second, extract the answer to this question from the text. The answer must be word for word exactly as it appears in the text.",
                f"You are a student and a teacher is teaching you about a new topic. Ask a short follow-up question about something the teacher hasn't mentioned yet at all. You must not ask something you already know the answer to from the teacher's explanations. You must not ask for further clarification if the teacher already mentioned something in passing. The question should be self-contained. It must not contain the word \"other\" as in \"which other\" or \"what other\". The question should start with one of {random.sample(question_words, 3)}"]

            prompt = TextLabelPrompt(
                input_variables=arguments.input_variables,
                target_variable=arguments.target_variable,
                task_description=task_descriptions[index],
            )

            raw_prompt = prompt.get_prompt_text(fewshot_examples)
            print(raw_prompt)

            current_unlabeled_examples = unlabeled_examples.select(range(i, min(len(unlabeled_examples), i + arguments.save_steps)))
            generated_dataset, original_dataset = generator.generate(
                support_examples=fewshot_examples,
                unlabeled_examples=current_unlabeled_examples,
                prompt_template=prompt,
                max_prompt_calls=arguments.max_prompt_calls,
                support_examples_per_prompt=arguments.support_examples_per_prompt,
                return_original_dataset=True,
            )

            generated_dataset = generated_dataset.map(split_columns)

            assert len(generated_dataset) == len(original_dataset)

            # filter bad samples from generated dataset
            if index == 0:  # answerable questions
                generated_dataset = postprocess_squad_format(generated_dataset, add_answer_start=True)
                indices_to_keep = \
                generated_dataset.map(lambda example, idx: {'idx': idx if example['answers']['answer_start'][0] >= 0 else -1},
                                      with_indices=True)['idx']
            else:  # unanswerable questions
                generated_dataset = postprocess_squad_format(generated_dataset, add_answer_start=False)
                indices_to_keep = \
                generated_dataset.map(lambda example, idx: {'idx': idx if example['answers']['answer_start'] == [] else -1},
                                      with_indices=True)['idx']
            indices_to_keep = [i for i in indices_to_keep if i != -1]

            generated_dataset = generated_dataset.select(indices_to_keep)
            original_dataset = original_dataset.select(indices_to_keep)

            # add id and title to generated dataset
            generated_dataset = generated_dataset.add_column("id", original_dataset['id'])
            generated_dataset = generated_dataset.add_column("title", original_dataset['title'])
            generated_dataset = generated_dataset.remove_columns("qa")

            ids_to_keep = set(original_dataset['id'])
            original_dataset = dataset.filter(lambda example: example['id'] in ids_to_keep)

            features = generated_dataset.features
            features["answers"] = Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int64', id=None)}, length=-1, id=None)
            generated_dataset = generated_dataset.cast(features)

            filtered_generated_datasets.append(generated_dataset)
            filtered_original_datasets.append(original_dataset)

            filtered_generated_concatenated_dataset = concatenate_datasets(filtered_generated_datasets)
            filtered_generated_concatenated_dataset.save_to_disk(f"{dataset_name}-generated-{index}-{i * arguments.save_steps}")
            filtered_original_concatenated_dataset = concatenate_datasets(filtered_original_datasets)
            filtered_original_concatenated_dataset.save_to_disk(f"{dataset_name}-original-{index}-{i * arguments.save_steps}")
    if arguments.push_to_hub:
        filtered_generated_concatenated_dataset.push_to_hub(f"{dataset_name}-generated-{len(filtered_generated_concatenated_dataset)}", private=False)
        filtered_original_concatenated_dataset.push_to_hub(f"{dataset_name}-original-{len(filtered_original_concatenated_dataset)}", private=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default="text-davinci-003")
    parser.add_argument("--max_generation_length", type=int, default=100)
    parser.add_argument(
        "--task_description",
        type=str,
        default="Given a context and a question, generate an answer that occurs exactly and only once in the text.",
    )
    parser.add_argument("--dataset", type=str, default="squad_v2")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--input_variables", type=str, nargs="+", default=["context"])
    parser.add_argument("--target_variable", type=str, default="qa")
    parser.add_argument("--output_format", type=str, default="text")
    parser.add_argument("--max_prompt_calls", type=int, default=20)
    parser.add_argument("--support_examples_per_prompt", type=int, default=1)
    parser.add_argument("--push_to_hub", action="store_false") #TODO set default back to store_true
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--num_labels", type=int, default=10)
    args = parser.parse_args()
    run(args)
