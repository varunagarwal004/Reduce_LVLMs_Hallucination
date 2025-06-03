# Results Analysis Script

This script analyzes the results generated from the visual puzzles and object detection evaluation scripts. It provides comprehensive metrics, statistical analysis, and visualizations to help understand the performance of different reasoning strategies.

## Features

- **Automatic Task Detection**: Detects whether results are from visual puzzles or object detection tasks
- **Strategy Comparison**: Compares self-verification, chain-of-thought, and direct approaches
- **Statistical Testing**: Performs Wilcoxon signed-rank tests for significance
- **Hallucination Analysis**: Special metrics for object detection (false positive/negative rates)
- **Visualizations**: Generates comprehensive plots and charts
- **Summary Reports**: Creates detailed markdown reports with key findings

## Usage

### Help Command

```bash
python scripts/analyze_results.py --help
```

### Basic Analysis

```bash
python scripts/analyze_results.py analyze --results-dir results_puzzles
```

### Full Analysis with Visualizations and Report

```bash
python scripts/analyze_results.py analyze \
    --results-dir results_puzzles \
    --output-dir analysis_output \
    --visualizations \
    --report
```

### Object Detection Analysis

```bash
python scripts/analyze_results.py analyze \
    --results-dir results_pope \
    --output-dir pope_analysis \
    --visualizations \
    --report
```

### Using Short Flags

```bash
python scripts/analyze_results.py analyze \
    -r results_puzzles \
    -o analysis_output \
    -v \
    --report
```

## Command Line Arguments

- `--results-dir` / `-r`: Directory containing CSV result files (required)
- `--output-dir` / `-o`: Directory to save analysis outputs (default: "analysis_output")
- `--visualizations` / `-v`: Generate visualization plots
- `--report`: Generate summary report in markdown format

## Input File Format

The script expects CSV files with specific column structures:

### Chain of Thought Results

- `question`: The question asked
- `answer`: Ground truth answer
- `cot_reasoning`: Chain of thought reasoning
- `cot_answer`: Final answer after CoT
- `direct_answer`: Direct answer without CoT
- `cot_correct`: Boolean indicating if CoT answer is correct
- `direct_correct`: Boolean indicating if direct answer is correct

### Self-Verification Results

- `question`: The question asked
- `answer`: Ground truth answer
- `initial_reasoning`: Initial reasoning
- `initial_answer`: Initial answer
- `verification_response`: Verification step response
- `final_answer`: Final answer after verification
- `direct_answer`: Direct answer without verification
- `initial_correct`: Boolean indicating if initial answer is correct
- `verified_correct`: Boolean indicating if verified answer is correct
- `direct_correct`: Boolean indicating if direct answer is correct

### Object Detection Results (Additional Columns)

- `false_positive`: Boolean indicating hallucination (claiming object exists when it doesn't)
- `false_negative`: Boolean indicating miss (claiming object doesn't exist when it does)

## Output Files

### Visualizations (when `--visualizations` flag is used)

- `accuracy_comparison.png`: Bar chart comparing accuracy across models and strategies
- `strategy_improvement.png`: Improvement of strategies over direct approach
- `hallucination_metrics.png`: Object detection specific metrics (for POPE results)

### Report (when `--report` flag is used)

- `summary_report.md`: Comprehensive markdown report with:
  - Basic accuracy metrics
  - Strategy comparisons
  - Statistical significance tests
  - Hallucination metrics (for object detection)
  - Key findings and recommendations

## Metrics Calculated

### Basic Metrics

- **Accuracy**: Percentage of correct answers
- **Count**: Number of examples evaluated

### Object Detection Specific Metrics

- **Hallucination Rate**: Percentage of false positives
- **Miss Rate**: Percentage of false negatives
- **Precision**: 1 - hallucination_rate
- **Recall**: 1 - miss_rate
- **F1 Score**: Harmonic mean of precision and recall

### Statistical Tests

- **Wilcoxon Signed-Rank Test**: Compares paired samples between strategies
- **p-value**: Statistical significance threshold (< 0.05 considered significant)

## Example Workflow

1. **Run Evaluations**: First run the evaluation scripts to generate results

   ```bash
   # Visual puzzles
   python scripts/visual_puzzles.py --model llava --strategy both --amount 100

   # Object detection
   python scripts/object_detection.py --model llava --strategy all --amount 100
   ```

2. **Analyze Results**: Run the analysis script

   ```bash
   python scripts/analyze_results.py analyze \
       --results-dir results_puzzles \
       --visualizations \
       --report
   ```

3. **Review Outputs**: Check the generated visualizations and summary report in the output directory

## Key Features

### Automatic Strategy Detection

The script automatically detects and handles different evaluation strategies:

- Self-verification (sv)
- Chain of thought (cot)
- Direct evaluation (baseline)

### Statistical Significance

Performs pairwise comparisons between strategies using non-parametric tests to determine if improvements are statistically significant.

### Comprehensive Reporting

Generates detailed reports including:

- Best performing strategies per model
- Overall performance rankings
- Strategy improvements over baseline
- Statistical significance of improvements

### Flexible Input Handling

Works with results from different models and evaluation amounts, automatically parsing filenames to extract metadata.

## Troubleshooting

### Common Issues

1. **No CSV files found**: Ensure the results directory contains CSV files from the evaluation scripts
2. **Missing columns**: Verify that CSV files have the expected column structure
3. **Import errors**: Install required dependencies using the requirements file
4. **Visualization errors**: Ensure matplotlib backend is properly configured

### File Naming Convention

The script expects CSV files to follow this naming pattern:
`{model}_{amount}_{strategy}_results.csv`

Examples:

- `llava_100_sv_results.csv`
- `gemma3_100_cot_results.csv`
- `openai_50_sv_results.csv`
