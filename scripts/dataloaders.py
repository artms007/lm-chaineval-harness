from abc import ABC, abstractmethod
import json
from datasets import load_dataset


# =====================
# Base Class
# =====================

class DataLoader(ABC):
    """An abstract base class for all data loaders to enforce the implementation of the load method."""
    @abstractmethod
    def load(self) -> list[dict]:
        pass


# =====================
# Testing Code
# =====================

class TestDataLoader(DataLoader):
    """A test data loader that returns a predefined list of dictionaries based on the specified dataset number."""

    def __init__(self, dataset_args: dict = None):
        self.dataset_args = dataset_args if dataset_args is not None else {}
        self.dataset_num = self.dataset_args.get('num', 2)  # Default number of test data is 2

    def load(self) -> list[dict]:
        test_data = []
        for i in range(1, self.dataset_num + 1):
            test_data.append({
                "task_id": f"test_{i}",
                "prompt": f"test_prompt_{i}",
                "canonical_solution": f"test_solution_{i}",
                "test": f"test_test_{i}",
                "entry_point": f"test_entry_{i}"
            })
        return test_data


# =====================
# JSON Data Loader
# =====================

class JSONDataLoader(DataLoader):
    """A data loader for JSONL files."""
    def __init__(self, dataset_path: str, dataset_args: dict = None):
        self.dataset_path = dataset_path
        self.dataset_args = dataset_args if dataset_args is not None else {}
        self.dataset_num = self.dataset_args.get('num')

    def load(self) -> list[dict]: 
        dataset = []
        try:
            with open(self.dataset_path, 'r') as f:
                dataset = [json.loads(line.strip()) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.dataset_path} does not exist.")
        dataset = dataset if self.dataset_num is None else dataset[:self.dataset_num]
        return dataset


# =====================
# HuggingFace Data Loader
# =====================

class HFDataLoader(DataLoader):
    """A data loader for datasets available through the Hugging Face datasets library."""
    def __init__(self, dataset_path: str, dataset_args: dict = None):
        self.dataset_path = dataset_path
        self.dataset_args = dataset_args if dataset_args is not None else {}
        self.dataset_num = self.dataset_args.get('num')

    def load(self) -> list[dict]:
        split = self.dataset_args.get("split", "test")
        subset = self.dataset_args.get("subset")

        if subset:
            dataset = load_dataset(self.dataset_path, subset, split=split)
        else:
            dataset = load_dataset(self.dataset_path, split=split)
        
        dataset = [{k: v for k, v in item.items()} for item in dataset]
        dataset = dataset if self.dataset_num is None else dataset[:self.dataset_num]
        return dataset


# =====================
# Data Loader Factory
# =====================

class DataLoaderFactory:
    """Factory class to create appropriate data loader instances based on the provided dataset path."""
    @staticmethod
    def create(dataset_path: str, dataset_args: dict = None) -> DataLoader:
        if dataset_path == "test":
            return TestDataLoader(dataset_args)
        elif dataset_path.endswith(".jsonl"):
            return JSONDataLoader(dataset_path, dataset_args)
        else:
            return HFDataLoader(dataset_path, dataset_args)



# =====================
# Utility Function
# =====================

def load_testdata(dataset_path: str, dataset_args: dict = None):
    loader = DataLoaderFactory.create(dataset_path, dataset_args)
    return loader.load()


# =====================
# Base Class
# =====================

class DataLoader(ABC):
    """An abstract base class for all data loaders to enforce the implementation of the load method."""
    
    def __init__(self, dataset_path: str, args: dict):
        self.dataset_path = dataset_path
        self.args = args
        self.head = args['dataset_head|head']

    def __repr__(self):
        return self.dataset_path
    
    @abstractmethod
    def load(self) -> list[dict]:
        pass

# =====================
# Testing Code
# =====================

class TestDataLoader(DataLoader):
    """A test data loader that returns a predefined list of dictionaries based on the specified dataset number."""

    # def __init__(self, dataset_path: str, args: dict):
    #     super().__init__(dataset_path, args)

    def load(self) -> list[dict]:
        test_data = []
        n = self.head or 10
        for i in range(1, n + 1):
            test_data.append({
                "task_id": f"test_{i}",
                "prompt": f"test_prompt_{i}",
                "canonical_solution": f"test_solution_{i}",
                "test": f"test_test_{i}",
                "entry_point": f"test_entry_{i}"
            })
        return test_data



class JSONDataLoader(DataLoader):
    """A data loader for JSONL files."""
    # def __init__(self, dataset_path: str, args):
    #     self.dataset_path = dataset_path
    #     self.args = args
    #     self.head = args['dataset_head|head']

    def load(self) -> list[dict]: 
        dataset = []
        try:
            with open(self.dataset_path, 'r') as f:
                dataset = [json.loads(line.strip()) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.dataset_path} does not exist.")
        dataset = dataset if self.head is None else dataset[:self.head]
        return dataset


class HFDataLoader(DataLoader):
    """A data loader for datasets available through the Hugging Face datasets library."""

    def load(self) -> list[dict]:
        split = self.args["split|=test"]
        subset = self.args["subset"]

        if subset:
            dataset = load_dataset(self.dataset_path, subset, split=split)
        else:
            dataset = load_dataset(self.dataset_path, split=split)
        
        dataset = [{k: v for k, v in item.items()} for item in dataset]
        dataset = dataset if self.head is None else dataset[:self.head]
        return dataset

def load_testdata(args):
    dataset_path = args['dataset_path|dataset']
    if dataset_path is None:
        return TestDataLoader('dataset/dummy', args).load()
    elif dataset_path.endswith(".jsonl"):
        return JSONDataLoader(dataset_path, args).load()
    else:
        return HFDataLoader(dataset_path, args).load()
