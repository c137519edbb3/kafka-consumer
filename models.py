from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io
import base64
import numpy as np

class CLIPInferenceModel:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.candidate_labels = []  # Will be set later

    def set_labels(self, labels):
        """Update the candidate labels for classification"""
        self.candidate_labels = labels

    def preprocess_image(self, image_data):
        """Convert base64 string to PIL Image"""
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        return image

    def predict(self, image_data):
        """Run inference on the image"""
        image = self.preprocess_image(image_data)
        inputs = self.processor(
            text=self.candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Return predictions as a list of (label, probability) tuples
        return [(label, prob.item()) for label, prob in zip(self.candidate_labels, probs[0])]

