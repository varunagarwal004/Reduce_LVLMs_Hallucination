import base64
import io
from abc import ABC, abstractmethod

import torch
from datasets import Dataset
from PIL import Image


class BaseLVLMModel(ABC):
    def __init__(
        self,
        model_name: str,
        hf_token: str,
        system_prompt: str,
        prompt_prefix: str,
        prompt_suffix: str,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.model_name = model_name
        self.hf_token = hf_token

        self._load_model()
        self._load_processor()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _load_processor(self):
        pass

    def process_image(self, image: Image.Image) -> str | None:
        """
        Process the image and return the base64 encoded image.
        Args:
            image (Image.Image): The image to process.
        Returns:
            The base64 encoded image.
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, image.format if image.format else "JPEG")
            im = buffer.getvalue()
            base64_image = base64.b64encode(im).decode("utf-8")
            return base64_image
        except Exception as e:
            print(f"Error: {e}")
            print(image.format, image.format_description)
            return None

    @abstractmethod
    def format_messages(
        self,
        image: Image.Image,
        prompt: str,
        options: list[str] | None = None,
        use_prefix_suffix: bool | None = None,
    ) -> list[dict]:
        pass

    @abstractmethod
    def generate_response(
        self,
        image: Image.Image,
        question: str,
        options: list[str] | None = None,
        use_prefix_suffix: bool | None = None,
    ) -> str | None:
        pass

    @abstractmethod
    def generate_response_batch(
        self,
        images: list[Image.Image],
        questions: list[str],
        options: list[list[str]] | None = None,
        use_prefix_suffix: bool | None = None,
    ) -> list[str]:
        pass

    def evaluate_dataset(
        self,
        dataset: Dataset,
        amount: int = 100,
        batch_size: int = 1,
        rand: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ) -> tuple[list[bool], list[str]]:
        """
        Evaluate the model on a dataset.
        Args:
            dataset (Dataset): The dataset to evaluate on.
            amount (int): The number of datapoints to evaluate on.
            batch_size (int): The batch size to use.
            rand (bool): Whether to shuffle the dataset.
            seed (int): The seed to use for shuffling.
            verbose (bool): Whether to print the results.
        Returns:
            tuple[list[bool], list[str]]: The results and responses.
        """
        if rand:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(amount))

        images = dataset["image"]
        questions = dataset["question"]
        options = dataset["options"]
        answers = dataset["answer"]

        results = []
        responses = []

        for i in range(0, amount, batch_size):
            batch_end = min(i + batch_size, amount)
            batch_images = images[i:batch_end]
            batch_questions = questions[i:batch_end]
            batch_options = options[i:batch_end]
            batch_answers = answers[i:batch_end]

            batch_responses = self.generate_response_batch(
                batch_images, batch_questions, batch_options
            )

            for j, response in enumerate(batch_responses):
                correct = response.strip().upper() == batch_answers[j].strip().upper()
                results.append(correct)
                responses.append(response)

                if verbose:
                    print(f"Question: {batch_questions[j]}")
                    print(f"Response: {response}")
                    print(f"Answer: {batch_answers[j]}")
                    print(f"Correct: {correct}\n")

        return results, responses
