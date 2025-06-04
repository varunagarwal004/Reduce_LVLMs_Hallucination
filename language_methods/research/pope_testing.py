# %%
import os

# %%
import torch
from datasets import load_dataset

from lvlm_models.chain_of_thought import ChainOfThoughtLVLM
from lvlm_models.llava import LlavaModel
from lvlm_models.self_verification import SelfVerificationLVLM

torch.cuda.empty_cache()
# %%
system_prompt = (
    "<role>\n"
    "You are a helpful visual assistant that can accurately identify "
    "objects in images.\n"
    "</role>"
)

prompt_prefix = (
    "Look at the image carefully and determine whether the object mentioned "
    "in the question is present in the image."
)

prompt_suffix = "Answer with only 'yes' or 'no', nothing else."
model = LlavaModel(
    model_name="llava-hf/llava-1.5-7b-hf",
    hf_token=os.getenv("HF_TOKEN"),
    system_prompt=system_prompt,
    prompt_prefix=prompt_prefix,
    prompt_suffix=prompt_suffix,
)
# %%
cot_model = ChainOfThoughtLVLM(model, cot_strategy="yes_no_object")
# %%
dataset = load_dataset("lmms-lab/POPE", split="test")
# %%
cot_model.generate_response_cot(dataset["image"][0], dataset["question"][0])
# %%
self_verification = SelfVerificationLVLM(
    model,
    reasoning_strategy="yes_no_object",
)
# %%
self_verification.generate_response_with_verification(dataset["image"][0], dataset["question"][0])
# %%
cot_model.evaluate_dataset_with_cot("lmms-lab/POPE", amount=100, verbose=True)
# %%
