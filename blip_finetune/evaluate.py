import nltk
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image

nltk.download("punkt")
nltk.download("punkt_tab")

def generate_caption(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def evaluate_bleu_on_test_dataset(test_dataset, model, processor, config):
    images_folder = config["images_folder"]
    bleu_scores = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}
    for example in test_dataset:
        image_path = example["image_path"]
        real_caption = example["label"]
        generated_caption = generate_caption(image_path, model, processor)
        reference = [real_caption]
        bleu_1 = sentence_bleu(reference, generated_caption.split(), weights=(1, 0, 0, 0))
        bleu_2 = sentence_bleu(reference, generated_caption.split(), weights=(0.5, 0.5, 0, 0))
        bleu_3 = sentence_bleu(reference, generated_caption.split(), weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = sentence_bleu(reference, generated_caption.split(), weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores["bleu_1"].append(bleu_1)
        bleu_scores["bleu_2"].append(bleu_2)
        bleu_scores["bleu_3"].append(bleu_3)
        bleu_scores["bleu_4"].append(bleu_4)
    for metric, scores in bleu_scores.items():
        print(f"
