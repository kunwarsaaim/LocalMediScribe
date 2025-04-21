import logging
from typing import Dict, Generator, Union

import ollama

logger = logging.getLogger(__name__)


class OllamaService:
    """
    A service class for interacting with Ollama models.
    """

    def __init__(self, model_name: str = "mistral"):
        """
        Initialize the Ollama service.

        Args:
            model_name: The name of the Ollama model to use
        """
        self.model_name = model_name
        logger.info(f"Initializing Ollama service with model: {model_name}")

    def load_model(self) -> None:
        """
        Load the specified Ollama model.
        """
        logger.info(f"Loading Ollama model: {self.model_name}")
        ollama.pull(self.model_name)
        logger.info(f"Model {self.model_name} loaded successfully")

    def unload_model(self):
        """Release model resources to free memory"""
        # Release Ollama model resources if applicable
        # You might need to call any shutdown/cleanup API methods if available

        # If you have a client instance or model reference, set it to None
        if hasattr(self, "client"):
            self.client = None

        # Force garbage collection
        import gc

        gc.collect()

        return True

    def generate_note(
        self,
        transcript: str,
        system_prompt: str = "You are a medical assistant. Generate a structured medical note based on this transcript.",
        stream: bool = True,
    ) -> Union[Dict, Generator[Dict, None, None]]:
        """
        Generate a note from a transcript using the Ollama model.

        Args:
            transcript: The transcript text to process
            system_prompt: The system prompt to guide the model's behavior
            stream: If True, returns a stream of response chunks

        Returns:
            Either the full response dictionary or a generator yielding response chunks
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": transcript,
            },
        ]

        logger.info(f"Generating note using {self.model_name}")
        return ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=stream,
        )
