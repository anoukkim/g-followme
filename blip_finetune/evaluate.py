import nltk
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image

nltk.download('punkt')

def generate_caption(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def calculate_bleu_scores(reference, generated):
    reference = [nltk.word_tokenize(ref) for ref in reference]
    generated = nltk.word_tokenize(generated)
    bleu_1 = sentence_bleu(reference, generated, weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu(reference, generated, weights=(0.5, 0.5, 0, 0))
    bleu_3 = sentence_bleu(reference, generated, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = sentence_bleu(reference, generated, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu_1, bleu_2, bleu_3, bleu_4

def evaluate_bleu_on_test_dataset(test_dataset, model, processor, config):
    bleu_scores = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}
    for example in test_dataset:
        image_path = f"images/{example['filename']}"
        real_caption = example['label']
        generated_caption = generate_caption(image_path, model, processor)
        scores = calculate_bleu_scores([real_caption], generated_caption)
        for i, score in enumerate(scores):
            bleu_scores[f"bleu_{i+1}"].append(score)

    avg_scores = {key: sum(values) / len(values) for key, values in bleu_scores.items()}
    for metric, score in avg_scores.items():
        print(f"Average {metric} score: {score:.4f}")