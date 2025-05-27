# %%
import os

from datasets import load_dataset
from dotenv import load_dotenv

from lvlm_models.openai import OpenAIVisionModel

load_dotenv("../.env")
# %%
model = OpenAIVisionModel(
    model_name="gpt-4.1-mini-2025-04-14",
    api_key=os.getenv("OPENAI_API_KEY"),
    system_prompt=(
        "<role>\n"
        "You are a helpful visual assistant that can solve visual puzzles and "
        "reasoning tasks.\n"
        "</role>"
    ),
    prompt_prefix=("Analyze this visual puzzle in the image and answer the following question:"),
    prompt_suffix=(
        "Choose the best answer from the provided options.\n"
        "Respond only with the letter of the correct option, no extra text, formatting, or "
        "punctuation. JUST THE LETTER, NO EXTRA TEXT."
    ),
)
# %%
dataset = load_dataset("neulab/VisualPuzzles", split="train")
# %%
model.evaluate_dataset(
    dataset=dataset,
    amount=10,
    verbose=True,
)
# %%
dataset["question"][0]
# %%
dataset["options"][0]
# %%
dataset["answer"][0]
# %%
model.generate_response(
    image=dataset["image"][0],
    question=dataset["question"][0],
    options=dataset["options"][0],
)
# %%
