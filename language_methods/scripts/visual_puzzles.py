"""Script to run visual puzzle evaluations using different reasoning strategies."""

import gc
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import typer
from dotenv import load_dotenv

from lvlm_models.base_lvlm import BaseLVLMModel
from lvlm_models.chain_of_thought import ChainOfThoughtLVLM
from lvlm_models.gemma3 import Gemma3Model
from lvlm_models.llava import LlavaModel
from lvlm_models.openai import OpenAIVisionModel
from lvlm_models.self_verification import SelfVerificationLVLM

app = typer.Typer()


class Strategy(str, Enum):
    """Evaluation strategy options."""

    SELF_VERIFICATION = "sv"
    CHAIN_OF_THOUGHT = "cot"
    BOTH = "both"


def setup_model(model_name: str) -> BaseLVLMModel:
    """Initialize and return the base LlavaModel."""
    load_dotenv()
    if model_name == "llava":
        return LlavaModel(
            model_name="llava-hf/llava-1.5-7b-hf",
            hf_token=os.getenv("HF_TOKEN"),
            system_prompt=(
                "<role>\n"
                "You are a helpful visual assistant that can solve visual puzzles and "
                "reasoning tasks.\n"
                "</role>"
            ),
            prompt_prefix=(
                "Analyze this visual puzzle in the image and answer the following question:"
            ),
            prompt_suffix=(
                "Choose the best answer from the provided options.\n"
                "Respond only with the letter of the correct option, no extra text or punctuation."
            ),
        )
    elif model_name == "gemma3":
        return Gemma3Model(
            model_name="google/gemma-3-4b-it",
            hf_token=os.getenv("HF_TOKEN"),
            system_prompt=(
                "<role>\n"
                "You are a helpful visual assistant that can solve visual puzzles and "
                "reasoning tasks.\n"
                "</role>"
            ),
            prompt_prefix=(
                "Analyze this visual puzzle in the image and answer the following question:"
            ),
            prompt_suffix=(
                "Choose the best answer from the provided options.\n"
                "Respond only with the letter of the correct option, no extra text or punctuation."
            ),
        )
    elif model_name == "openai":
        return OpenAIVisionModel(
            model_name="gpt-4.1-mini-2025-04-14",
            api_key=os.getenv("OPENAI_API_KEY"),
            system_prompt=(
                "<role>\n"
                "You are a helpful visual assistant that can solve visual puzzles and "
                "reasoning tasks.\n"
                "</role>"
            ),
            prompt_prefix=(
                "Analyze this visual puzzle in the image and answer the following question:"
            ),
            prompt_suffix=(
                "Choose the best answer from the provided options.\n"
                "Respond only with the letter of the correct option, no extra text or punctuation."
            ),
        )
    else:
        raise ValueError(f"Model {model_name} not supported")


def cleanup_model(
    model: Optional[LlavaModel] = None,
    sv_model: Optional[SelfVerificationLVLM] = None,
    cot_model: Optional[ChainOfThoughtLVLM] = None,
) -> None:
    """Clean up models and free GPU memory."""
    if model:
        del model
    if sv_model:
        del sv_model
    if cot_model:
        del cot_model
    gc.collect()
    torch.cuda.empty_cache()


def run_self_verification(
    model: BaseLVLMModel,
    output_path: Path,
    split: str,
    amount: int,
    batch_size: int,
    rand: bool,
    seed: int,
    verbose: bool,
) -> None:
    """Run evaluation using self-verification strategy."""
    sv_model = SelfVerificationLVLM(
        base_model=model,
        reasoning_strategy="visual_reasoning",
    )

    results_df = sv_model.evaluate_dataset_with_sv(
        dataset_name="neulab/VisualPuzzles",
        split=split,
        amount=amount,
        batch_size=batch_size,
        rand=rand,
        seed=seed,
        verbose=verbose,
    )

    results_df.to_csv(
        output_path / f"{model.model_name.split('/')[-1]}_{amount}_sv_results.csv", index=False
    )
    cleanup_model(sv_model=sv_model)


def run_chain_of_thought(
    model: BaseLVLMModel,
    output_path: Path,
    split: str,
    amount: int,
    rand: bool,
    seed: int,
    verbose: bool,
) -> None:
    """Run evaluation using chain-of-thought strategy."""
    cot_model = ChainOfThoughtLVLM(
        base_model=model,
        cot_strategy="visual_puzzle",
    )

    results_df = cot_model.evaluate_dataset_with_cot(
        dataset_name="neulab/VisualPuzzles",
        split=split,
        amount=amount,
        rand=rand,
        seed=seed,
        verbose=verbose,
    )

    results_df.to_csv(
        output_path / f"{model.model_name.split('/')[-1]}_{amount}_cot_results.csv", index=False
    )
    cleanup_model(cot_model=cot_model)


@app.command()
def evaluate(
    model_name: str = typer.Option(
        "llava",
        "--model",
        "-m",
        help="Model to use",
    ),
    strategy: Strategy = typer.Option(
        Strategy.BOTH,
        "--strategy",
        "-s",
        help="Evaluation strategy to use (sv, cot, or both)",
    ),
    output_dir: str = typer.Option(
        "results_puzzles",
        "--output-dir",
        "-o",
        help="Directory to save results",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Dataset split to use",
    ),
    amount: int = typer.Option(
        100,
        "--amount",
        "-n",
        help="Number of examples to evaluate",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Batch size for evaluation (only used for self-verification)",
    ),
    random: bool = typer.Option(
        True,
        "--random",
        "-r",
        help="Whether to shuffle the dataset",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for shuffling",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Whether to print verbose output",
    ),
) -> None:
    """Run visual puzzle evaluations using specified strategy."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = setup_model(model_name)

    try:
        if strategy in (Strategy.SELF_VERIFICATION, Strategy.BOTH):
            typer.echo("Running self-verification evaluation...")
            run_self_verification(
                model=model,
                output_path=output_path,
                split=split,
                amount=amount,
                batch_size=batch_size,
                rand=random,
                seed=seed,
                verbose=verbose,
            )

        if strategy in (Strategy.CHAIN_OF_THOUGHT, Strategy.BOTH):
            typer.echo("Running chain-of-thought evaluation...")
            run_chain_of_thought(
                model=model,
                output_path=output_path,
                split=split,
                amount=amount,
                rand=random,
                seed=seed,
                verbose=verbose,
            )

        typer.echo(f"Results saved to {output_path}")

    finally:
        cleanup_model(model=model)


if __name__ == "__main__":
    app()
