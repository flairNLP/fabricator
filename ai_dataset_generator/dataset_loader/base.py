class BaseDatasetLoader:
    def __init__(self, dataset_path, dataset_name, dataset_type):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type

    def load(self):
        raise NotImplementedError


from datasets import load_dataset


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    def __init__(self, dataset_path, dataset_name, dataset_type):
        super().__init__(dataset_path, dataset_name, dataset_type)
        self.dataset = load_dataset(dataset_name, dataset_type)

    def load(self):
        return self.dataset
