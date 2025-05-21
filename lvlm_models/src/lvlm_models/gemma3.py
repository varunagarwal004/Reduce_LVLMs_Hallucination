import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from lvlm_models.base_lvlm import BaseLVLMModel


class Gemma3Model(BaseLVLMModel):
    def _load_model(self):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            token=self.hf_token,
            device_map=self.device,
        ).eval()

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def format_messages(
        self,
        image: Image.Image,
        prompt: str,
        options: list[str] | None = None,
        use_prefix_suffix: bool | None = None,
    ) -> list[dict]:
        """
        Format the messages for the Gemma3 model.
        Args:
            image (Image.Image): The image to process.
            prompt (str): The prompt to use.
            options (list[str] | None): The options to use. Default is None.
        Returns:
            The formatted messages.
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (f"{self.prompt_prefix}\n{prompt}\n{self.prompt_suffix}\n")
                        if use_prefix_suffix
                        else prompt,
                    },
                ],
            },
        ]

        if options is not None:
            options_str = [
                f"{letter}: {option}" for letter, option in zip(["A", "B", "C", "D"], options)
            ]
            messages[-1]["content"][-1]["text"] += (
                "<options>\n"
                f"These are the options to choose from: {' | '.join(options_str)}\n"
                "</options>"
            )
        return messages

    def generate_response(
        self,
        image: Image.Image,
        question: str,
        options: list[str] | None = None,
        use_prefix_suffix: bool | None = None,
    ) -> str | None:
        """
        Generate text from the image and prompt using Gemma3.
        Args:
            image (Image.Image): The image to process.
            question (str): The prompt to use.
            options (list[str] | None): The options to use. Default is None.
        Returns:
            The generated text.
        """
        messages = self.format_messages(image, question, options, use_prefix_suffix)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=1000, do_sample=False)
            generation = outputs[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def generate_response_batch(
        self,
        images: list[Image.Image],
        questions: list[str],
        options: list[list[str]] | None = None,
        use_prefix_suffix: bool | None = None,
    ) -> list[str]:
        """
        Generate responses for a batch of images and questions efficiently with Gemma3.
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
            msg = self.format_messages(image, question, opts, use_prefix_suffix)
            messages.append(msg)

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
            generation = output[input_lens[i] :]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            decoded_responses.append(decoded)

        return decoded_responses
