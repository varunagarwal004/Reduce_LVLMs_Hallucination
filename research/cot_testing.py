# %%
get_ipython().run_line_magic("load_ext", "autoreload")  # noqa
get_ipython().run_line_magic("autoreload", "2")  # noqa
# %%
import gc
import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv

from lvlm_models.chain_of_thought import ChainOfThoughtLlava
from lvlm_models.llava import LlavaModel

# %%
load_dotenv("../.env")

model = LlavaModel(
    model_name="llava-hf/llava-1.5-7b-hf",
    hf_token=os.getenv("HF_TOKEN"),
    system_prompt=(
        "You are a helpful visual assistant that can solve visual puzzles and reasoning tasks."
    ),
    prompt_prefix="Analyze this visual puzzle in the image and answer the following question:",
    prompt_suffix=(
        "Choose the best answer from the provided options.\n"
        "Respond only with the letter of the correct option, no extra text or punctuation."
    ),
)
# %%
cot_model = ChainOfThoughtLlava(
    base_model=model,
    cot_strategy="visual_puzzle",
)
# %%
results_df = cot_model.evaluate_dataset_with_cot(
    dataset_name="neulab/VisualPuzzles",
    split="train",
    amount=100,
    verbose=True,
)
# %%
puzzles = load_dataset("neulab/VisualPuzzles", split="train")
# %%
puzzles[:10]
# %%
results_df.to_csv("llava_cot_puzzles_2.csv", index=False)
# %%
images = puzzles["image"][:2]
questions = puzzles["question"][:2]
options = puzzles["options"][:2]
answers = puzzles["answer"][:2]
# %%
results = cot_model.generate_response_cot(
    image=images[0],
    question=questions[0],
    options=options[0],
)
results
# %%
for i, result in enumerate(results):
    print(f"Question: {questions[i]}")
    print(f"Answer: {result[1]}")
    print(f"Ground Truth: {answers[i]}")
    print(f"Reasoning: {result[0]}")
    print("-" * 100)
# %%
model.generate_response_batch(
    images=images,
    questions=questions,
    options=options,
)
# %%
images[0]
# %%
cot_model.generate_response_cot(
    image=images[0],
    question=questions[0],
    options=options[0],
)
# %%


del model
del cot_model
gc.collect()
torch.cuda.empty_cache()
# %%
