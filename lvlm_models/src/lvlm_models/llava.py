import base64
import io

import torch
from datasets import Dataset
from PIL import Image
from transformers.models.llava import LlavaForConditionalGeneration
from transformers.models.llava.processing_llava import LlavaProcessor


class LlavaModel:
    def __init__(
        self,
        model_name: str,
        hf_token: str,
        system_prompt: str,
        prompt_prefix: str,
        prompt_suffix: str,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            token=hf_token,
            device_map=self.device,
        ).eval()

        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.system_prompt = system_prompt
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

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

    def format_messages(
        self, image: Image.Image, prompt: str, options: list[str] | None = None
    ) -> list[dict]:
        """
        Format the messages for the VLLM.
        Args:
            image (Image.Image): The image to process.
            prompt (str): The prompt to use.
            options (list[str] | None): The options to use. Default is None.
        Returns:
            The formatted messages.
        """
        base64_image = self.process_image(image)
        if base64_image is None:
            return None
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_image},
                    {
                        "type": "text",
                        "text": (f"{self.prompt_prefix}\n{prompt}\n{self.prompt_suffix}\n"),
                    },
                ],
            },
        ]
        if options is not None:
            options_str = [
                f"{letter}: {option}" for letter, option in zip(["A", "B", "C", "D"], options)
            ]
            messages[-1]["content"][-1]["text"] += (
                f"These are the options to choose from: {' | '.join(options_str)}"
            )
        return messages

    def generate_response(
        self, image: Image.Image, question: str, options: list[str] | None = None
    ) -> str | None:
        """
        Generate text from the image and prompt.
        Args:
            image (Image.Image): The image to process.
            prompt (str): The prompt to use.
            options (list[str] | None): The options to use. Default is None.
        Returns:
            The generated text.
        """
        messages = self.format_messages(image, question, options)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            # Output contains the entire sequence (system and user messages), so we need to slice
            # it to get the generation
            generation = outputs[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def generate_response_batch(
        self,
        images: list[Image.Image],
        questions: list[str],
        options: list[list[str]] | None = None,
    ) -> list[str]:
        """
        Generate responses for a batch of images and questions efficiently.
        Args:
            images (list[Image.Image]): The images to process.
            questions (list[str]): The questions to use.
            options (list[list[str]] | None): The options to use. Default is None.
        Returns:
            list[str]: The generated responses, one per input.
        """
        messages = []
        for i, (image, question) in enumerate(zip(images, questions)):
            opts = options[i] if options is not None else None
            msg = self.format_messages(image, question, opts)
            if msg is None:  # Handle potential image processing failures
                raise ValueError(f"Failed to process image at index {i}")
            messages.append(msg)

        # We need padding and truncation for batched processing
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            truncation=True,
            max_length=2048,  # Set a reasonable max_length to control memory usage
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        # For batched input with padding, we need to know where each sequence starts
        # This is the length of input_ids for each sample in the batch
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        decoded_responses = []
        for i, output in enumerate(outputs):
            # Extract only the generated part for each sample
            generation = output[input_lens[i] :]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            decoded_responses.append(decoded)

        return decoded_responses

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
                correct = response.strip().upper() == batch_answers[i + j].strip().upper()
                results.append(correct)
                responses.append(response)

                if verbose:
                    print(f"Question: {batch_questions[j]}")
                    print(f"Response: {response}")
                    print(f"Correct: {correct}\n")

        return results, responses
