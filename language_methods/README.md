# Reduce LVLMs Hallucination

A research project focused on understanding and reducing hallucinations in Large Vision-Language Models (LVLMs) to improve factual accuracy and reliability in multimodal tasks.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Installing uv

```bash
curl -fsSL https://astral.sh/uv/install.sh | bash
```

Or if you're using pip:

```bash
pip install uv
```

### Setting up the project

1. Clone the repository:

```bash
git clone <repository-url>
cd Reduce_LVLMs_Hallucination/language_methods
```

2. Install dependencies:

```bash
uv sync
```

This will install all dependencies defined in the pyproject.toml file, including the local `lvlm_models` package.

## Project Structure

### LVLM Models

The `lvlm_models` package contains implementations of various Large Vision-Language Models and reasoning methods:

- `base_lvlm.py`: Defines the base LVLM interface for all models
- `llava.py`: Implementation of the LLaVA model
- `openai.py`: Interface for OpenAI's multimodal models
- `gemma3.py`: Implementation of Google's Gemma 3 model

### Reasoning Methods

Two main reasoning methods are implemented to reduce hallucinations:

#### Chain of Thought (CoT)

Located in `chain_of_thought.py`, this method improves reasoning by guiding the model through a step-by-step thinking process before reaching a final answer. CoT helps models break down complex visual tasks into logical steps, reducing the likelihood of hallucinations.

#### Self-Verification

Located in `self_verification.py`, this technique enables models to verify their own responses by critically evaluating their answers. The model first generates an answer, then assesses its confidence and factual accuracy, and finally revises the answer if necessary.

## Scripts

All scripts support the `--help` flag to display available options and their descriptions.

```bash
python -m scripts/visual_puzzles.py --help
```

The `scripts/` directory contains various utilities for running experiments and analyzing results:

### Visual Puzzles Evaluation

```bash
python -m scripts/visual_puzzles.py --model llava --strategy cot
```

Key options:

- `--model` / `-m`: Model to use (llava, gemma3, openai)
- `--strategy` / `-s`: Evaluation strategy (sv, cot, both)
- `--amount` / `-n`: Number of examples to evaluate
- `--output-dir` / `-o`: Directory to save results (default: results_puzzles)

### Object Detection Evaluation

```bash
python -m scripts/object_detection.py --model llava --strategy sv
```

Key options:

- `--model` / `-m`: Model to use (llava, gemma3, openai)
- `--strategy` / `-s`: Evaluation strategy (sv, cot, all)
- `--amount` / `-n`: Number of examples to evaluate per category
- `--output-dir` / `-o`: Directory to save results (default: results_pope)

### Results Analysis

```bash
python -m scripts/analyze_results.py --results-dir results_puzzles --output-dir analysis_puzzles
```

Key options:

- `--results-dir` / `-r`: Directory containing CSV result files
- `--output-dir` / `-o`: Directory to save analysis outputs
- `--visualizations` / `-v`: Generate visualization plots
- `--report`: Generate summary report

### Interactive Demo

```bash
python -m streamlit run scripts/demo.py
```

Provides an interactive interface to test different models and reasoning methods on custom images.
The demo can run in cpu but CUDA is recommended.

## Results

Experimental results are stored in the `results_puzzles/` and `results_pope/` directories, with corresponding analysis in `analysis_puzzles/` and `analysis_pope/`.
