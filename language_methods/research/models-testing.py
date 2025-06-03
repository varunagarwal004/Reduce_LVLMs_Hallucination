# %%
import base64
import gc
import json
import os
import random
from io import BytesIO
from typing import Any

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from IPython.display import display
from PIL import Image
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForVision2Seq
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.gemma3 import Gemma3ForConditionalGeneration
from transformers.models.instructblip import (
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)
from transformers.models.llava import LlavaForConditionalGeneration
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

# %%
load_dotenv("../.env")
hf_token = os.getenv("HUGGINGFACE_TOKEN", "")

models = {
    "gemma-3-4b-it": {
        "name": "google/gemma-3-4b-it",
        "model_class": Gemma3ForConditionalGeneration,
        "processor": AutoProcessor,
    },
    "gemma-3-1b-it": {
        "name": "google/gemma-3-1b-it",
        "model_class": Gemma3ForConditionalGeneration,
        "processor": AutoProcessor,
    },
    "internvl-1-5-2b": {
        "name": "OpenGVLab/Mini-InternVL-Chat-2B-V1-5",
        "model_class": AutoModel,
        "processor": AutoProcessor,
    },
    "internvl-1-5-4b": {
        "name": "OpenGVLab/Mini-InternVL-Chat-4B-V1-5",
        "model_class": AutoModel,
        "processor": AutoProcessor,
    },
    "internvl-2-5-2b": {
        "name": "OpenGVLab/InternVL2_5-2B",
        "model_class": AutoModel,
        "processor": AutoProcessor,
    },
    "internvl-2-5-4b": {
        "name": "OpenGVLab/InternVL2_5-4B",
        "model_class": AutoModel,
        "processor": AutoProcessor,
    },
    "llava-1-5-7b": {
        "name": "llava-hf/llava-1.5-7b-hf",
        "model_class": LlavaForConditionalGeneration,
        "processor": AutoProcessor,
    },
    "instructblip-7b": {
        "name": "Salesforce/instructblip-vicuna-7b",
        "model_class": InstructBlipForConditionalGeneration,
        "processor": InstructBlipProcessor,
    },
    "smolvlm": {
        "name": "HuggingFaceTB/SmolVLM-Instruct",
        "model_class": AutoModelForVision2Seq,
        "processor": AutoProcessor,
    },
    "qwen-7b-vl": {
        "name": "Qwen/Qwen2-VL-7B-Instruct",
        "model_class": Qwen2VLForConditionalGeneration,
        "processor": AutoProcessor,
    },
}


# %%
def load_model_and_processor(model: dict[str, Any], hf_token: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model["name"]
    model_class = model["model_class"]
    processor_class = model["processor"]
    model = model_class.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        load_in_8bit=True,
    ).eval()

    processor = processor_class.from_pretrained(model_name, use_fast=True)

    return model, processor


# %%
def process_image(image: Image.Image):
    try:
        buffer = BytesIO()
        image.save(buffer, image.format if image.format else "JPEG")
        im = buffer.getvalue()
        base64_image = base64.b64encode(im).decode("utf-8")
        return base64_image
    except Exception as e:
        print(f"Error: {e}")
        print(image.format, image.format_description)
        return None


# %%
def run_dataset_check(
    model: torch.nn.Module,
    processor: AutoProcessor,
    dataset: Dataset,
    user_prompt_prefix: str = "",
    user_prompt_suffix: str = "",
    amount: int = 100,
    rand: bool = False,
    seed: int = 42,
    verbose: bool = False,
):
    results: list[bool] = []
    responses: list[str] = []
    if rand:
        random.seed(seed)
        subset_indices = random.sample(range(len(dataset)), amount)
    else:
        subset_indices = range(amount)
    for i in subset_indices:
        datapoint = dataset[i]
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": process_image(datapoint["image"])},
                    {
                        "type": "text",
                        "text": (
                            f"{user_prompt_prefix} \n"
                            f"{datapoint['question']} \n"
                            f"{user_prompt_suffix} "
                        ),
                    },
                ],
            },
        ]
        if "options" in datapoint.keys() and datapoint["options"] is not None:
            options = [
                f"{letter}: {option}"
                for letter, option in zip(["A", "B", "C", "D"], datapoint["options"])
            ]
            messages[-1]["content"][-1]["text"] += (
                f"These are the options to choose from: {' | '.join(options)}"
            )

        if verbose:
            print(f"Prompt: {messages[-1]['content'][-1]['text']}")
            display(datapoint["image"])

        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)
        except Exception as e:
            print(f"Error: {e}:\n{json.dumps(messages, indent=2)}")
            decoded = ""
            continue

        if (
            datapoint["answer"].lower() in decoded.lower()
            or decoded.lower() in datapoint["answer"].lower()
        ):
            if verbose:
                print(f"Correct: model {decoded.lower()} - dataset {datapoint['answer']}\n\n")
            results.append(True)
        else:
            if verbose:
                print(f"Incorrect: model {decoded.lower()} - dataset {datapoint['answer']}\n\n")
            results.append(False)

        responses.append(decoded.lower())

    return results, responses


# %%
selected_model = None
selected_processor = None
if selected_model is not None and selected_processor is not None:
    del selected_model
    del selected_processor
    gc.collect()
    torch.cuda.empty_cache()
# %%
selected_model, selected_processor = load_model_and_processor(models["instructblip-7b"], hf_token)
# %%
puzzles = load_dataset("neulab/VisualPuzzles")
# %%
results_puzzle, responses_puzzle = run_dataset_check(
    model=selected_model,
    processor=selected_processor,
    dataset=puzzles["train"],
    user_prompt_suffix=(
        "Select the correct option from those provided by the puzzle or in this statement."
    ),
    user_prompt_prefix=(
        "Respond only with the letter of the correct option, no extra text or punctuation."
    ),
    amount=100,
    rand=True,
    seed=42,
    verbose=True,
)
# %%
correct_rate = sum(results_puzzle) / len(results_puzzle)
print(f"Correct rate: {correct_rate}")
# %%
