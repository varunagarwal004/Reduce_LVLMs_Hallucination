import os

from datasets import load_dataset

from models.chain_of_thought import ChainOfThoughtLlava
from models.llava import LlavaModel


def main():
    """
    Example demonstrating the Chain of Thought reasoning for visual puzzle questions.
    """
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set the HF_TOKEN environment variable")

    # Initialize LlavaModel
    model = LlavaModel(
        model_name="llava-hf/llava-1.5-7b-hf",
        hf_token=hf_token,
        system_prompt=(
            "You are a helpful visual assistant that can solve visual puzzles and reasoning tasks."
        ),
        prompt_prefix="Analyze this visual puzzle and answer the following question:",
        prompt_suffix="Choose the best answer from the provided options.",
    )

    # Initialize Chain of Thought wrapper with visual puzzle strategy
    puzzle_cot_model = ChainOfThoughtLlava(
        base_model=model,
        cot_strategy="visual_puzzle",
    )

    try:
        dataset = load_dataset("neulab/VisualPuzzles", split="train")
        print("Loaded visual puzzle dataset successfully.")
    except Exception as e:
        print(f"Could not load the requested dataset: {e}")
        print("Using a fallback dataset for demonstration...")
        dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation[:10]")

    # Example 1: Solve a single puzzle
    example_idx = 0
    image = dataset[example_idx]["image"]
    question = dataset[example_idx]["question"]

    # Options might be structured differently based on dataset
    if "choices" in dataset[example_idx]:
        options = dataset[example_idx]["choices"]
    elif "options" in dataset[example_idx]:
        options = dataset[example_idx]["options"]
    else:
        # Fallback - this will vary depending on dataset structure
        options = ["Option A", "Option B", "Option C", "Option D"]

    # Get ground truth if available
    answer = dataset[example_idx].get("answer", None)

    print(f"Visual Puzzle Question: {question}")
    if answer:
        print(f"Ground Truth Answer: {answer}")
    print(f"Options: {options}")

    print("\nSolving with Chain of Thought...")

    # Get answer using the structured format
    puzzle_reasoning, puzzle_answer = puzzle_cot_model.generate_response_cot(
        image=image, question=question, options=options
    )

    # Check if the model followed the FINAL ANSWER format
    format_followed = "FINAL ANSWER:" in puzzle_reasoning

    # Display results
    print("\n--- Visual Puzzle CoT Results ---")
    print(f"Answer: {puzzle_answer}")
    print(f"Model followed FINAL ANSWER format: {format_followed}")

    # Print the relevant portion near the answer
    print("\nReasoning excerpt (focusing on conclusion):")
    print_reasoning_conclusion(puzzle_reasoning)


def print_reasoning_conclusion(reasoning, context_lines=5):
    """Print the conclusion part of the reasoning with focus on the FINAL ANSWER section."""
    if "FINAL ANSWER:" in reasoning:
        # Extract the part with the FINAL ANSWER
        parts = reasoning.split("FINAL ANSWER:")
        before = parts[0]
        after = parts[1]

        # Get a few lines before FINAL ANSWER
        before_lines = before.split("\n")
        context_before = (
            "\n".join(before_lines[-context_lines:])
            if len(before_lines) > context_lines
            else before
        )

        # Get a few lines after FINAL ANSWER
        after_lines = after.split("\n")
        context_after = (
            "\n".join(after_lines[:context_lines]) if len(after_lines) > context_lines else after
        )

        print(f"{context_before}\nFINAL ANSWER:{context_after}")
    else:
        # Just print the last part of the reasoning
        lines = reasoning.split("\n")
        print("\n".join(lines[-10:]))


def print_reasoning_excerpt(reasoning, max_length=500):
    """Print a readable excerpt of the reasoning."""
    if len(reasoning) > max_length:
        # Print first and last parts of the reasoning
        first_part = reasoning[: max_length // 2]
        last_part = reasoning[-max_length // 2 :]
        print(f"{first_part}\n[...]\n{last_part}")
    else:
        print(reasoning)


if __name__ == "__main__":
    main()
