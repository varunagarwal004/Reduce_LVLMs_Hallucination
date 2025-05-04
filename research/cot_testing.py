# %%
import os

from datasets import load_dataset
from dotenv import load_dotenv

from lvlm_models.src.lvlm_models.chain_of_thought import ChainOfThoughtLlava
from lvlm_models.src.lvlm_models.llava import LlavaModel

# %%
load_dotenv("../.env")

model = LlavaModel(
    model_name="llava-hf/llava-1.5-7b-hf",
    hf_token=os.getenv("HF_TOKEN"),
    system_prompt=(
        "You are a helpful visual assistant that can solve visual puzzles and reasoning tasks."
    ),
    prompt_prefix="Analyze this visual puzzle and answer the following question:",
    prompt_suffix="Choose the best answer from the provided options.",
)
# %%
cot_model = ChainOfThoughtLlava(
    model=model,
    cot_strategy="visual_puzzle",
)
# %%
puzzles = load_dataset("neulab/VisualPuzzles", split="train")
# %%
puzzles[0]
# %%
image = [puzzle["image"] for puzzle in puzzles[:10]]
question = [puzzle["question"] for puzzle in puzzles[:10]]
options = [puzzle["options"] for puzzle in puzzles[:10]]
answer = [puzzle.get("answer", None) for puzzle in puzzles[:10]]
# %%
results = cot_model.generate_response_cot_batch(
    images=image,
    questions=question,
    options=options,
)
# %%
for i, result in enumerate(results):
    print(f"Question: {question[i]}")
    print(f"Answer: {result[1]}")
    print(f"Ground Truth: {answer[i]}")
    print(f"Reasoning: {result[0]}")
    print("-" * 100)
# %%
