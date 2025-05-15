"""
File: core/utils/embedding_utils.py
Description: Utility functions for generating text and image embeddings.
"""
from transformers import CLIPProcessor, CLIPModel
import torch
from openai import OpenAI
from PIL import Image
from config.config import Config
import torch.nn as nn

clip_model = CLIPModel.from_pretrained(Config.IMAGE_EMBEDDING_MODEL)
clip_processor = CLIPProcessor.from_pretrained(Config.IMAGE_EMBEDDING_MODEL)
client = OpenAI()

text_projection = nn.Linear(1536, Config.SINGLE_EMBEDDING_DIM)

def get_text_embedding(text, model=None):
    """
    Generates text embeddings using OpenAI's API or CLIP model and reduces dimensionality if needed.

    Args:
        text (str): Input text.
        model (str, optional): Model to use for generating embeddings.

    Returns:
        list: Text embedding vector as a list of floats.
    """
    model = model or Config.TEXT_EMBEDDING_MODEL
    text = text.replace("\n", " ") or "this is blank"

    if model == "openai/clip-vit-large-patch14":
        inputs = clip_processor(text=[text], return_tensors="pt")
        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)
        return embedding.squeeze(0).tolist()
    else:
        response = client.embeddings.create(input=[text], model=model)
        embedding = torch.tensor(response.data[0].embedding)
        reduced_embedding = text_projection(embedding)
        return reduced_embedding.tolist()

def get_image_embedding(image_path):
    """
    Generates image embeddings using CLIP.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list: Image embedding vector as a list of floats.
    """
    image = Image.open(image_path).convert("RGB")
    
    inputs = clip_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    
    return embedding.squeeze(0).tolist()

def generate_combined_embedding(text, image_path):
    """
    Generates a combined embedding for a text and an image.

    Args:
        text (str): Text to be embedded.
        image_path (str): Path to the image.

    Returns:
        dict: Dictionary containing 'combined_embedding' and optionally 'text_embedding' and 'image_embedding' if configured.
    """
    text_embedding = get_text_embedding(text)
    image_embedding = get_image_embedding(image_path)
    

    text_tensor = torch.tensor(text_embedding)
    image_tensor = torch.tensor(image_embedding)

    text_tensor = text_tensor / text_tensor.norm()
    image_tensor = image_tensor / image_tensor.norm()

    image_tensor = image_tensor * Config.IMAGE_EMBEDDING_WEIGHT
    
    combined_embedding = torch.cat((text_tensor, image_tensor), dim=0)
    
    result = {"combined_embedding": combined_embedding.tolist()}
    
    if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
        result["text_embedding"] = text_tensor.tolist()
        result["image_embedding"] = image_tensor.tolist()
    
    return result