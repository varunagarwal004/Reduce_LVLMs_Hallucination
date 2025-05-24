import os
from typing import List, Optional

from openai import OpenAI
from PIL import Image

from lvlm_models.base_lvlm import BaseLVLMModel


class OpenAIVisionModel(BaseLVLMModel):
    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        system_prompt: str = "",
        prompt_prefix: str = "",
        prompt_suffix: str = "",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Call the parent initialization
        super().__init__(
            model_name=model_name,
            hf_token="",  # Not used for OpenAI
            system_prompt=system_prompt,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
        )

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def _load_model(self):
        # No local model to load for API-based approach
        pass

    def _load_processor(self):
        # No processor to load for API-based approach
        pass

    def format_messages(
        self,
        image: Image.Image,
        prompt: str,
        options: List[str] = None,
        use_prefix_suffix: bool = None,
    ) -> List[dict]:
        """
        Format the messages for the OpenAI Vision API.
        Args:
            image (Image.Image): The image to process.
            prompt (str): The prompt to use.
            options (List[str]): The options to use. Default is None.
            use_prefix_suffix (bool): Whether to use the prefix and suffix. Default is None.
        Returns:
            The formatted messages.
        """
        # Encode image to base64
        base64_image = self.process_image(image)

        # Format the prompt text with prefix/suffix if needed
        formatted_prompt = prompt
        if use_prefix_suffix:
            formatted_prompt = f"{self.prompt_prefix}\n{prompt}\n{self.prompt_suffix}\n"

        # Add options if provided
        if options is not None:
            options_str = [
                f"{letter}: {option}" for letter, option in zip(["A", "B", "C", "D"], options)
            ]
            formatted_prompt += (
                "<options>\n"
                f"These are the options to choose from: {' | '.join(options_str)}\n"
                "</options>"
            )

        # Build message structure for OpenAI Vision API
        messages = []

        # Add system message if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add user message with image and text
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        )

        return messages

    def generate_response(
        self,
        image: Image.Image,
        question: str,
        options: List[str] = None,
        use_prefix_suffix: bool = None,
    ) -> Optional[str]:
        """
        Generate text from the image and prompt using OpenAI Vision API.
        Args:
            image (Image.Image): The image to process.
            question (str): The prompt to use.
            options (List[str]): The options to use. Default is None.
            use_prefix_suffix (bool): Whether to use the prefix and suffix. Default is None.
        Returns:
            The generated text.
        """
        messages = self.format_messages(image, question, options, use_prefix_suffix)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, max_tokens=1000
            )
            input_cost = 0.4e-6 * response.usage.prompt_tokens
            output_cost = 1.6e-6 * response.usage.completion_tokens
            print(f"Total cost: {(input_cost + output_cost):.6f} USD")

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None

    def generate_response_batch(
        self,
        images: List[Image.Image],
        questions: List[str],
        options: List[List[str]] = None,
        use_prefix_suffix: bool = None,
    ) -> List[str]:
        """
        Generate responses for a batch of images and questions using OpenAI Vision API.
        Note: This makes separate API calls for each image as OpenAI doesn't support true batching.
        Args:
            images (List[Image.Image]): The images to process.
            questions (List[str]): The questions to use.
            options (List[List[str]]): The options to use. Default is None.
            use_prefix_suffix (bool): Whether to use the prefix and suffix. Default is None.
        Returns:
            List[str]: The generated responses, one per input.
        """
        responses = []

        for i, (image, question) in enumerate(zip(images, questions)):
            opts = options[i] if options is not None else None
            response = self.generate_response(image, question, opts, use_prefix_suffix)
            responses.append(response if response else "")

        return responses
