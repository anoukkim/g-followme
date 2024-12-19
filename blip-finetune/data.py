from datasets import Dataset
import pandas as pd

def preprocess_and_save(label_csv, images_folder, save_path):
    df = pd.read_csv(label_csv)
    df["image_path"] = df["image"].apply(lambda x: f"{images_folder}/{x}")
    df = df[["image_path", "label"]]
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(save_path)

def load_and_split_dataset(preprocessed_dataset_path):
    dataset = Dataset.load_from_disk(preprocessed_dataset_path)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    val_test_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]
    return train_dataset, val_dataset, test_dataset
