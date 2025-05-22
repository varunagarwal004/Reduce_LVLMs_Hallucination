# %%
get_ipython().run_line_magic("load_ext", "autoreload")  # noqa
get_ipython().run_line_magic("autoreload", "2")  # noqa
# %%
import gc
import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv

from lvlm_models.gemma3 import Gemma3Model
from lvlm_models.llava import LlavaModel
from lvlm_models.self_verification import SelfVerificationLVLM

# %%
load_dotenv("../.env")

model = LlavaModel(
    model_name="llava-hf/llava-1.5-7b-hf",
    hf_token=os.getenv("HF_TOKEN"),
    system_prompt=(
        "<role>\n"
        "You are a helpful visual assistant that can solve visual puzzles and reasoning tasks.\n"
        "</role>"
    ),
    prompt_prefix="Analyze this visual puzzle in the image and answer the following question:",
    prompt_suffix=(
        "Choose the best answer from the provided options.\n"
        "Respond only with the letter of the correct option, no extra text or punctuation."
    ),
)
# %%
sv_model = SelfVerificationLVLM(
    base_model=model,
    reasoning_strategy="visual_reasoning",
)
# %%
results_df = sv_model.evaluate_dataset_with_sv(
    dataset_name="neulab/VisualPuzzles",
    split="train",
    amount=100,
    verbose=True,
)
# %%
results_df.to_csv("llava_sv_puzzles.csv", index=False)
# %%
puzzles = load_dataset("neulab/VisualPuzzles", split="train")
# %%
puzzles[:10]
# %%
images = puzzles["image"][:2]
questions = puzzles["question"][:2]
options = puzzles["options"][:2]
answers = puzzles["answer"][:2]
# %%
result = sv_model.generate_response_with_verification(
    image=images[0],
    question=questions[0],
    options=options[0],
)
result
# %%
# Print detailed info for a single example
initial_reasoning, verification, final_answer = result
print(f"Question: {questions[0]}")
print(f"Initial Reasoning:\n{initial_reasoning}")
print(f"Verification:\n{verification}")
print(f"Final Answer: {final_answer}")
print(f"Ground Truth: {answers[0]}")
print(f"Correct: {sv_model.match_multiple_choice_answer(final_answer, answers[0])}")
print("-" * 100)
# %%
# Generate responses for multiple examples and print results
for i in range(2):
    initial_reasoning, verification, final_answer = sv_model.generate_response_with_verification(
        image=images[i],
        question=questions[i],
        options=options[i],
    )
    print(f"Question: {questions[i]}")
    print(f"Final Answer: {final_answer}")
    print(f"Ground Truth: {answers[i]}")
    print(f"Correct: {sv_model.match_multiple_choice_answer(final_answer, answers[i])}")
    print("-" * 100)
# %%
# Test direct response (without verification)
direct_responses = model.generate_response_batch(
    images=images,
    questions=questions,
    options=options,
)
direct_responses
# %%
# Compare direct vs verified responses
for i in range(len(direct_responses)):
    print(f"Question: {questions[i]}")
    print(f"Direct Response: {direct_responses[i]}")
    initial_reasoning, verification, final_answer = sv_model.generate_response_with_verification(
        image=images[i],
        question=questions[i],
        options=options[i],
    )
    print(f"Verified Response: {final_answer}")
    print(f"Ground Truth: {answers[i]}")
    print(
        f"Direct Correct: {sv_model.match_multiple_choice_answer(direct_responses[i], answers[i])}"
    )
    print(f"Verified Correct: {sv_model.match_multiple_choice_answer(final_answer, answers[i])}")
    print("-" * 100)
# %%
# Cleanup
del model
del sv_model
gc.collect()
torch.cuda.empty_cache()
# %%
