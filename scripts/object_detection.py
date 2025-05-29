"""Script to evaluate object hallucination using the POPE dataset."""

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
    ALL = "all"


def setup_model(model_name: str) -> BaseLVLMModel:
    """Initialize and return the specified model."""
    load_dotenv()

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

    if model_name == "llava":
        return LlavaModel(
            model_name="llava-hf/llava-1.5-7b-hf",
            hf_token=os.getenv("HF_TOKEN"),
            system_prompt=system_prompt,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
        )
    elif model_name == "gemma3":
        return Gemma3Model(
            model_name="google/gemma-3-4b-it",
            hf_token=os.getenv("HF_TOKEN"),
            system_prompt=system_prompt,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
        )
    elif model_name == "openai":
        return OpenAIVisionModel(
            model_name="gpt-4.1-mini-2025-04-14",
            api_key=os.getenv("OPENAI_API_KEY"),
            system_prompt=system_prompt,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")


def cleanup_model(
    model: Optional[BaseLVLMModel] = None,
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
        reasoning_strategy="yes_no_object",
    )

    results_df = sv_model.evaluate_dataset_with_sv(
        dataset_name="lmms-lab/POPE",
        split=split,
        amount=amount,
        batch_size=batch_size,
        rand=rand,
        seed=seed,
        verbose=verbose,
    )

    # Add additional metrics specific to object hallucination
    results_df["false_positive"] = (results_df["final_answer"].str.strip().str.lower() == "yes") & (
        results_df["answer"].str.strip().str.lower() == "no"
    )
    results_df["false_negative"] = (results_df["final_answer"].str.strip().str.lower() == "no") & (
        results_df["answer"].str.strip().str.lower() == "yes"
    )

    hallucination_rate = results_df["false_positive"].mean()
    miss_rate = results_df["false_negative"].mean()

    print(f"Accuracy: {results_df['verified_correct'].mean():.4f}")
    print(f"Hallucination rate: {hallucination_rate:.4f}")
    print(f"Miss rate: {miss_rate:.4f}")

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
        cot_strategy="object_verification",
    )

    results_df = cot_model.evaluate_dataset_with_cot(
        dataset_name="lmms-lab/POPE",
        split=split,
        amount=amount,
        rand=rand,
        seed=seed,
        verbose=verbose,
    )

    # Add additional metrics specific to object hallucination
    results_df["false_positive"] = (results_df["cot_answer"].str.strip().str.lower() == "yes") & (
        results_df["answer"].str.strip().str.lower() == "no"
    )
    results_df["false_negative"] = (results_df["cot_answer"].str.strip().str.lower() == "no") & (
        results_df["answer"].str.strip().str.lower() == "yes"
    )

    hallucination_rate = results_df["false_positive"].mean()
    miss_rate = results_df["false_negative"].mean()

    print(f"Accuracy: {results_df['cot_correct'].mean():.4f}")
    print(f"Hallucination rate: {hallucination_rate:.4f}")
    print(f"Miss rate: {miss_rate:.4f}")

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
        help="Model to use (llava, gemma3, openai)",
    ),
    strategy: Strategy = typer.Option(
        Strategy.ALL,
        "--strategy",
        "-s",
        help="Evaluation strategy to use (direct, sv, cot, or all)",
    ),
    output_dir: str = typer.Option(
        "results_pope",
        "--output-dir",
        "-o",
        help="Directory to save results",
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="Dataset split to use",
    ),
    amount: int = typer.Option(
        100,
        "--amount",
        "-n",
        help="Number of examples to evaluate (max per category)",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Batch size for evaluation (only used for direct evaluation)",
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
    """Run POPE object hallucination evaluations using specified strategy."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = setup_model(model_name)

    try:
        if strategy in (Strategy.SELF_VERIFICATION, Strategy.ALL):
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

        if strategy in (Strategy.CHAIN_OF_THOUGHT, Strategy.ALL):
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
