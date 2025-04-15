import os
import torch
import json
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class OpenSourceWebGenerator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Initialize the web generator with an open-source model.

        Args:
            model_name (str): Hugging Face model name (default: 'mistralai/Mistral-7B-Instruct-v0.1')
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.model.to(self.device)

        # Setup text generation pipeline
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def generate_web_content(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> dict:
        """
        Generate web content using an open-source model.

        Args:
            prompt (str): The input prompt.
            temperature (float): Controls randomness (0.0 to 1.0).
            max_tokens (int): Maximum number of tokens to generate.

        Returns:
            dict: Generated content.
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        response = self.generator(prompt, max_length=max_tokens, temperature=temperature, do_sample=True)

        content = response[0]["generated_text"]
        return {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "model": self.model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }

# Example usage
if __name__ == "__main__":
    generator = OpenSourceWebGenerator()  # Uses 'mistralai/Mistral-7B-Instruct-v0.1'
    prompt = "Write an engaging introduction to artificial intelligence."
    
    result = generator.generate_web_content(prompt)
    print(json.dumps(result, indent=4))
