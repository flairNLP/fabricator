from pathlib import Path

from haystack.nodes import FARMReader
from datasets import load_dataset
import json

from haystack.utils import SquadData


def hf_to_squad(dataset_name: str) -> str:
    dataset = load_dataset(dataset_name, split="train").shuffle().to_dict()
    paragraphs = []

    for i in range(len(dataset["context"])):
        if dataset["answers"][i]["text"]:
            answers = [
                {"text": dataset["answers"][i]["text"][0], "answer_start": dataset["answers"][i]["answer_start"][0]}]
            impossible = False
        else:
            answers = []
            impossible = True
        paragraph = {"qas": [{"id": dataset["id"][i], "question": dataset["question"][i], "answers": answers,
                              "is_impossible": impossible}], "context": dataset["context"][i]}
        paragraphs.append(paragraph)

    squad = {"version": "1.0", "data": [{"title": "test", "paragraphs": paragraphs}]}

    filename = f"squad-{dataset_name.split('/')[-1]}"
    with open(f"{filename}.json", "w") as f:
        f.write(json.dumps(squad))
    return filename


if __name__ == '__main__':
    config = [1_000, 10_000, 20_000]
    model_name = "roberta-base"  # "bert-base-uncased" is a worse alternative
    dataset_names = ["julianrisch/qa-dataset-generated-21020", "julianrisch/qa-dataset-original-21020"]

    for dataset_name in dataset_names:
        squad_filename = hf_to_squad(dataset_name)
        dataset = SquadData.from_file(filename=f"{squad_filename}.json")
        for num_samples in config:
            train_filename = f"{squad_filename}-{num_samples}.json"
            sample = dataset.sample_questions(num_samples)
            SquadData(squad_data=sample).save(train_filename)

            # Model Training
            reader_directory = f"{squad_filename}-{num_samples}-{model_name}"
            reader = FARMReader(model_name_or_path=model_name, return_no_answer=True, use_confidence_scores=False)
            reader.train(data_dir="..", train_filename=train_filename, dev_split=0.1, use_gpu=True, batch_size=16, max_seq_len=384)
            reader.save(Path(reader_directory))

            # Model Evaluation
            reader = FARMReader(reader_directory, return_no_answer=True, use_confidence_scores=False, max_seq_len=384)
            reader_eval_results = reader.eval_on_file(data_dir="..", test_filename="dev-v2.0.json")
            with open("log.txt", "a") as log_file:
                log_file.write(str(reader_directory)+'\n')
                log_file.write(str(reader_eval_results)+'\n')

