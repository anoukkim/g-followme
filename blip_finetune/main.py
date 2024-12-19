from data import preprocess_and_save, load_and_split_dataset
from model import initialize_model_and_processor
from train import train_model
from evaluate import evaluate_bleu_on_test_dataset
from utils import load_config

if __name__ == "__main__":
    config = load_config()
    preprocess_and_save(config["label_csv"], config["images_folder"], config["preprocessed_dataset"])
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(config["preprocessed_dataset"])
    model, processor = initialize_model_and_processor()
    train_model(model, processor, train_dataset, val_dataset, config)
    evaluate_bleu_on_test_dataset(test_dataset, model, processor, config)
