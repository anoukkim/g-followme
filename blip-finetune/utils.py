import json
import torch

def load_config(file_path="config.json"):
    with open(file_path, "r") as file:
        return json.load(file)

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    labels = torch.stack([torch.tensor(item["labels"]) for item in batch])
    pixel_values = torch.stack([torch.tensor(item["pixel_values"]) for item in batch])
    return {
        "input_ids": input_ids.squeeze(1),
        "attention_mask": attention_mask.squeeze(1),
        "labels": labels.squeeze(1),
        "pixel_values": pixel_values.squeeze(1),
    }
