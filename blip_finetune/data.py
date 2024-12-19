from datasets import Dataset

def load_and_split_dataset(dataset_path):
    dataset = Dataset.load_from_disk(dataset_path)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    val_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']
    return train_dataset, val_dataset, test_dataset