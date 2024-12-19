from transformers import BlipProcessor, BlipForConditionalGeneration

def initialize_model_and_processor():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return model, processor