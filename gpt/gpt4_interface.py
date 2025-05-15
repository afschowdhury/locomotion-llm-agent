"""
File: gpt/gpt4_interface.py
Description: Implements the GPTInterface class for interacting with the GPT API.
"""


import openai
import base64
import json
import os
from config.config import Config

class GPTInterface:
    """
    Interacts with the GPT API, including prompt handling and requests.

    Attributes:
        client: The OpenAI client for sending requests.
        model_name: The name of the GPT model.
    """
    def __init__(self, client, model_name=None):
        """
        Initializes the GPTInterface with a client and model name.

        Args:
            client: OpenAI client instance.
            model_name (str, optional): Name of the GPT model.
        """
        self.client = client
        self.model_name = model_name or Config.GPT_MODEL_NAME

    def encode_image(self, image_path):
        """
        Encode an image file to a base64 string.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The base64-encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def send_gpt_request(self, prompt, encoded_image, json_schema=None, 
                        max_tokens=None, temperature=None, top_p=1.0, 
                        frequency_penalty=0.0, presence_penalty=0.0):
        """
        Sends a GPT request with a prompt and an encoded image.
        
        Args:
            prompt (str): Text prompt for GPT.
            encoded_image (str): Base64-encoded image.
            json_schema (dict, optional): JSON schema for response validation.
            max_tokens (int, optional): Maximum number of tokens for the response.
            temperature (float, optional): Sampling temperature.
            top_p (float, optional): Top-p sampling.
            frequency_penalty (float, optional): Frequency penalty for repeated tokens.
            presence_penalty (float, optional): Presence penalty for new topics.
            system_prompt (str, optional): System prompt for GPT.

        Returns:
            dict: GPT API response.
        """
        max_tokens = max_tokens or Config.GPT_MAX_TOKENS
        temperature = temperature or Config.GPT_PERCEPTION_TEMPERATURE
        
        system_prompt = self.load_prompt(os.path.join(Config.PROMPT_DIR, "system_prompt.txt"))

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"}
                }
            ]}
        ]

        request_params = {
            "model": self.model_name,
            "messages": message,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        response_format = None
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "default_name"),
                    "strict": True, 
                    "schema": json_schema
                }
            }
            request_params["response_format"] = response_format
    
        response = self.client.chat.completions.create(**request_params)

        return response
    
    @staticmethod
    def load_prompt(prompt_path):
        """
        Loads the prompt from a file.

        Args:
            prompt_path (str): Path to the prompt file.

        Returns:
            str: The raw prompt as a string.
        """
        with open(prompt_path, "r") as file:
            return file.read()

    @staticmethod
    def generate_prompt(prompt, **variables):
        """
        Renders a prompt by replacing placeholders with variables.

        Args:
            prompt (str): The raw prompt string.
            variables (dict): Mapping of placeholders to their values.

        Returns:
            str: The rendered prompt.
        """
        try:
            return prompt.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing value for placeholder: {e}")
        
    @staticmethod
    def load_schema(schema_path):
        """
        Loads a JSON schema from a file.

        Args:
            schema_path (str): Path to the JSON schema file.

        Returns:
            dict: The loaded JSON schema.
        """
        with open(schema_path, "r") as file:
            return json.load(file)
