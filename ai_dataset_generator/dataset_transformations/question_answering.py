from datasets import Dataset


def preprocess_squad_format(dataset: Dataset) -> Dataset:
    def preprocess(example):
        if example["answer"]:
            example["answer"] = example["answer"].pop()
        else:
            example["answer"] = ""
        return example

    dataset = dataset.flatten().rename_column("answers.text", "answer").map(preprocess)
    return dataset


def postprocess_squad_format(dataset: Dataset, add_answer_start: bool = True) -> Dataset:
    if add_answer_start:
        dataset = dataset.map(calculate_answer_start)

    def unify_answers(example):
        example["answer"] = {"text": example["answer"], "start": example["answer_start"]}
        return example

    dataset = dataset.map(unify_answers).remove_columns("answer_start")
    return dataset


def calculate_answer_start(example):
    answer_start = example["context"].find(example["answer"])
    if answer_start < 0:
        print(
            f'Could not calculate the answer start because the context "{example["context"]}" '
            f'does not contain the answer "{example["answer"]}".'
        )
        answer_start = -1
    else:
        # check that the answer doesn't occur more than once in the context
        second_answer_start = example["context"].find(example["answer"], answer_start + 1)
        if second_answer_start >= 0:
            print("Could not calculate the answer start because the context contains the answer more than once.")
            answer_start = -1
        else:
            answer_start = answer_start
    example["answer_start"] = answer_start
    return example
