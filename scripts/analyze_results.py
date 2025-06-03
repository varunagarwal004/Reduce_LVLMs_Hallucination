"""Script to analyze results from visual puzzles and object detection evaluations."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from scipy import stats

app = typer.Typer()


class ResultsAnalyzer:
    """Analyzer for evaluation results from different reasoning strategies."""

    def __init__(self, results_dir: str):
        """
        Initialize the analyzer with a results directory.

        Args:
            results_dir: Directory containing CSV result files
        """
        self.results_dir = Path(results_dir)
        self.results = {}
        self.task_type = None

    def load_results(self) -> None:
        """Load all CSV result files from the results directory."""
        csv_files = list(self.results_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.results_dir}")

        for csv_file in csv_files:
            key = self._extract_key_from_filename(csv_file.name)
            df = pd.read_csv(csv_file)
            self.results[key] = df

            # Determine task type from columns
            if self.task_type is None:
                if "false_positive" in df.columns or "false_negative" in df.columns:
                    self.task_type = "object_detection"
                else:
                    self.task_type = "visual_puzzles"

        print(f"Loaded {len(self.results)} result files")
        print(f"Detected task type: {self.task_type}")

    def _extract_key_from_filename(self, filename: str) -> str:
        """Extract a readable key from the CSV filename."""
        # Remove .csv extension
        name = filename.replace(".csv", "")

        # Extract model name, amount, and strategy
        parts = name.split("_")
        if len(parts) >= 3:
            model = parts[0]
            amount = parts[1] if parts[1].isdigit() else "unknown"
            strategy = "_".join(parts[2:])
            return f"{model}_{strategy}_{amount}"

        return name

    def calculate_basic_metrics(self) -> pd.DataFrame:
        """Calculate basic accuracy metrics for all loaded results."""
        metrics_data = []

        # Add DINO-X and LLaVA combination data
        dinox_llava_data = {
            "model": "DINO-X_LLaVA",
            "strategy": "combined",
            "amount": "all",
            "accuracy": 0.786,
            "count": 1200,
        }
        # metrics_data.append(dinox_llava_data)
        dinox_llava_direct = {
            "model": "DINO-X_LLaVA",
            "strategy": "direct",
            "amount": "all",
            "accuracy": 0.776,
            "count": 1200,
        }
        # metrics_data.append(dinox_llava_direct)

        yolo_llava_data = {
            "model": "YOLOv8_LLaVA",
            "strategy": "combined",
            "amount": "all",
            "accuracy": 0.885,
            "count": 2999,
        }
        # metrics_data.append(yolo_llava_data)
        yolo_llava_direct = {
            "model": "YOLOv8_LLaVA",
            "strategy": "direct",
            "amount": "all",
            "accuracy": 0.776,
            "count": 2999,
        }
        # metrics_data.append(yolo_llava_direct)

        for key, df in self.results.items():
            model, strategy, amount = self._parse_key(key)

            if strategy == "sv":
                # Self-verification results
                if "initial_correct" in df:
                    initial_acc = df["initial_correct"].mean()
                elif "initial_answer" in df and "answer" in df:
                    # Create the initial_correct column
                    df["initial_correct"] = (
                        df["initial_answer"].str.strip().str.lower()
                        == df["answer"].str.strip().str.lower()
                    )
                    initial_acc = df["initial_correct"].mean()
                else:
                    initial_acc = 0

                if "verified_correct" in df:
                    verified_acc = df["verified_correct"].mean()
                elif "final_answer" in df and "answer" in df:
                    # Create the verified_correct column
                    df["verified_correct"] = (
                        df["final_answer"].str.strip().str.lower()
                        == df["answer"].str.strip().str.lower()
                    )
                    verified_acc = df["verified_correct"].mean()
                else:
                    verified_acc = 0

                if "direct_correct" in df:
                    direct_acc = df["direct_correct"].mean()
                elif "direct_answer" in df and "answer" in df:
                    # Create the direct_correct column
                    df["direct_correct"] = (
                        df["direct_answer"].str.strip().str.lower()
                        == df["answer"].str.strip().str.lower()
                    )
                    direct_acc = df["direct_correct"].mean()
                else:
                    direct_acc = 0

                metrics_data.append(
                    {
                        "model": model,
                        "strategy": "initial",
                        "amount": amount,
                        "accuracy": initial_acc,
                        "count": len(df),
                    }
                )
                metrics_data.append(
                    {
                        "model": model,
                        "strategy": "self_verification",
                        "amount": amount,
                        "accuracy": verified_acc,
                        "count": len(df),
                    }
                )
                metrics_data.append(
                    {
                        "model": model,
                        "strategy": "direct",
                        "amount": amount,
                        "accuracy": direct_acc,
                        "count": len(df),
                    }
                )

            elif strategy == "cot":
                # Chain of thought results
                if "cot_correct" in df:
                    cot_acc = df["cot_correct"].mean()
                elif "cot_answer" in df and "answer" in df:
                    # Create the cot_correct column
                    df["cot_correct"] = (
                        df["cot_answer"].str.strip().str.lower()
                        == df["answer"].str.strip().str.lower()
                    )
                    cot_acc = df["cot_correct"].mean()
                else:
                    cot_acc = 0

                if "direct_correct" in df:
                    direct_acc = df["direct_correct"].mean()
                elif "direct_answer" in df and "answer" in df:
                    # Create the direct_correct column
                    df["direct_correct"] = (
                        df["direct_answer"].str.strip().str.lower()
                        == df["answer"].str.strip().str.lower()
                    )
                    direct_acc = df["direct_correct"].mean()
                else:
                    direct_acc = 0

                metrics_data.append(
                    {
                        "model": model,
                        "strategy": "chain_of_thought",
                        "amount": amount,
                        "accuracy": cot_acc,
                        "count": len(df),
                    }
                )
                metrics_data.append(
                    {
                        "model": model,
                        "strategy": "direct",
                        "amount": amount,
                        "accuracy": direct_acc,
                        "count": len(df),
                    }
                )

        return pd.DataFrame(metrics_data)

    def calculate_hallucination_metrics(self) -> Optional[pd.DataFrame]:
        """Calculate hallucination-specific metrics for all results."""
        metrics_data = []

        # Add DINO-X and LLaVA combination data
        dinox_llava_data = {
            "model": "DINO-X_LLaVA",
            "strategy": "combined",
            "amount": "all",
            "accuracy": 0.786,
            "true_positives": 554,
            "false_positives": 211,
            "true_negatives": 389,
            "false_negatives": 46,
            "hallucination_rate": 211 / (211 + 389),  # FP / (FP + TN)
            "miss_rate": 46 / (46 + 554),  # FN / (FN + TP)
            "precision": 0.724,
            "recall": 0.923,
            "f1_score": 0.812,
            "count": 554 + 211 + 389 + 46,
        }
        # metrics_data.append(dinox_llava_data)
        yolo_llava_data = {
            "model": "YOLOv8_LLaVA",
            "strategy": "combined",
            "amount": "all",
            "accuracy": 0.885,
            "precision": 0.909,
            "recall": 0.856,
            "f1_score": 0.882,
            "true_positives": 1950,
            "false_positives": 195,
            "true_negatives": 524,
            "false_negatives": 330,
            "hallucination_rate": 195 / (195 + 524),  # FP / (FP + TN)
            "miss_rate": 330 / (330 + 1950),  # FN / (FN + TP)
            "count": 2999,
        }
        # metrics_data.append(yolo_llava_data)

        for key, df in self.results.items():
            model, strategy, amount = self._parse_key(key)

            # For object detection tasks with explicit hallucination columns
            if "false_positive" in df.columns and "false_negative" in df.columns:
                hallucination_rate = df["false_positive"].mean()
                miss_rate = df["false_negative"].mean()
                precision = 1 - hallucination_rate
                recall = 1 - miss_rate

            # For visual puzzles - treat incorrect answers as hallucinations
            else:
                # Determine the accuracy column based on the strategy
                if strategy == "sv":
                    accuracy_col = "verified_correct" if "verified_correct" in df else None
                    if accuracy_col is None and "initial_answer" in df and "answer" in df:
                        # Create the verified_correct column
                        df["verified_correct"] = (
                            df["final_answer"].str.strip().str.lower()
                            == df["answer"].str.strip().str.lower()
                        )
                        accuracy_col = "verified_correct"
                elif strategy == "cot":
                    accuracy_col = "cot_correct" if "cot_correct" in df else None
                    if accuracy_col is None and "cot_answer" in df and "answer" in df:
                        # Create the cot_correct column
                        df["cot_correct"] = (
                            df["cot_answer"].str.strip().str.lower()
                            == df["answer"].str.strip().str.lower()
                        )
                        accuracy_col = "cot_correct"
                else:
                    accuracy_col = None

                if accuracy_col is not None:
                    # Treat incorrect answers as hallucinations and compute metrics
                    hallucination_rate = 1 - df[accuracy_col].mean()
                    miss_rate = (
                        hallucination_rate  # In puzzles, incorrect answers can be considered both
                    )
                    precision = 1 - hallucination_rate
                    recall = 1 - miss_rate
                else:
                    # Skip if we can't determine accuracy
                    continue

            f1_score = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            )

            if strategy == "sv":
                strategy_name = "self_verification"
                accuracy = df["verified_correct"].mean() if "verified_correct" in df else 0
            elif strategy == "cot":
                strategy_name = "chain_of_thought"
                accuracy = df["cot_correct"].mean() if "cot_correct" in df else 0
            else:
                strategy_name = strategy
                accuracy = df.get("accuracy", 0)

            metrics_data.append(
                {
                    "model": model,
                    "strategy": strategy_name,
                    "amount": amount,
                    "accuracy": accuracy,
                    "hallucination_rate": hallucination_rate,
                    "miss_rate": miss_rate,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "count": len(df),
                }
            )

        return pd.DataFrame(metrics_data)

    def _parse_key(self, key: str) -> Tuple[str, str, str]:
        """Parse key into model, strategy, and amount components."""
        parts = key.split("_")
        if len(parts) >= 3:
            model = parts[0]
            strategy = parts[1]
            amount = parts[2]
        else:
            model = "unknown"
            strategy = "unknown"
            amount = "unknown"
        return model, strategy, amount

    def compare_strategies(self) -> pd.DataFrame:
        """Compare performance across different strategies."""
        basic_metrics = self.calculate_basic_metrics()

        if basic_metrics.empty:
            return pd.DataFrame()

        # Create a direct comparison value for DINO-X_LLaVA
        # This is needed to calculate improvement in the pivot table
        # dinox_llava_direct = {
        #     "model": "DINO-X_LLaVA",
        #     "strategy": "direct",
        #     "amount": "all",
        #     "accuracy": 0.776,  # Assuming baseline accuracy of 65% for direct approach
        #     "count": 1200,
        # }

        # yolo_llava_direct = {
        #     "model": "YOLOv8_LLaVA",
        #     "strategy": "direct",
        #     "amount": "all",
        #     "accuracy": 0.776,  # Assuming baseline accuracy of 65% for direct approach
        #     "count": 2999,
        # }
        # # Add the direct comparison row if DINO-X_LLaVA exists in the data
        # if basic_metrics[basic_metrics["model"] == "DINO-X_LLaVA"].shape[0] > 0:
        #     # Check if a direct entry already exists
        #     if (
        #         basic_metrics[
        #             (basic_metrics["model"] == "DINO-X_LLaVA")
        #             & (basic_metrics["strategy"] == "direct")
        #         ].shape[0]
        #         == 0
        #     ):
        #         # Append only if it doesn't exist
        #         basic_metrics = pd.concat(
        #             [basic_metrics, pd.DataFrame([dinox_llava_direct])], ignore_index=True
        #         )
        # if basic_metrics[basic_metrics["model"] == "YOLOv8_LLaVA"].shape[0] > 0:
        #     # Check if a direct entry already exists
        #     if (
        #         basic_metrics[
        #             (basic_metrics["model"] == "YOLOv8_LLaVA")
        #             & (basic_metrics["strategy"] == "direct")
        #         ].shape[0]
        #         == 0
        #     ):
        #         # Append only if it doesn't exist
        #         basic_metrics = pd.concat(
        #             [basic_metrics, pd.DataFrame([yolo_llava_direct])], ignore_index=True
        #         )

        # Pivot to compare strategies side by side
        comparison = basic_metrics.pivot_table(
            index=["model", "amount"], columns="strategy", values="accuracy", aggfunc="mean"
        ).reset_index()

        # Calculate improvements
        if "direct" in comparison.columns:
            for strategy in ["self_verification", "chain_of_thought", "combined"]:
                if strategy in comparison.columns:
                    comparison[f"{strategy}_improvement"] = (
                        comparison[strategy] - comparison["direct"]
                    )

        return comparison

    def statistical_significance_test(self) -> Dict[str, Dict[str, float]]:
        """Perform statistical significance tests between strategies."""
        results = {}

        for key, df in self.results.items():
            model, strategy, amount = self._parse_key(key)

            if strategy == "sv" and "verified_correct" in df and "direct_correct" in df:
                # Compare self-verification vs direct
                verified_scores = df["verified_correct"].astype(float)
                direct_scores = df["direct_correct"].astype(float)

                # Check if we have enough non-identical pairs for the test
                try:
                    # Use pandas to check equality safely regardless of data type
                    identical_pairs = (verified_scores == direct_scores).sum()
                    if (
                        len(verified_scores) - identical_pairs >= 6
                    ):  # Wilcoxon needs at least 6 non-zero differences
                        statistic, p_value = stats.wilcoxon(
                            verified_scores, direct_scores, alternative="two-sided"
                        )

                        results[f"{model}_sv_vs_direct"] = {
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                        }
                    else:
                        results[f"{model}_sv_vs_direct"] = {
                            "statistic": 0.0,
                            "p_value": 1.0,
                            "significant": False,
                            "note": "Insufficient non-identical pairs for test",
                        }
                except Exception as e:
                    results[f"{model}_sv_vs_direct"] = {
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                        "error": str(e),
                    }

            elif strategy == "cot" and "cot_correct" in df and "direct_correct" in df:
                # Compare chain of thought vs direct
                cot_scores = df["cot_correct"].astype(float)
                direct_scores = df["direct_correct"].astype(float)

                # Check if we have enough non-identical pairs for the test
                try:
                    # Use pandas to check equality safely regardless of data type
                    identical_pairs = (cot_scores == direct_scores).sum()
                    if (
                        len(cot_scores) - identical_pairs >= 6
                    ):  # Wilcoxon needs at least 6 non-zero differences
                        statistic, p_value = stats.wilcoxon(
                            cot_scores, direct_scores, alternative="two-sided"
                        )

                        results[f"{model}_cot_vs_direct"] = {
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                        }
                    else:
                        results[f"{model}_cot_vs_direct"] = {
                            "statistic": 0.0,
                            "p_value": 1.0,
                            "significant": False,
                            "note": "Insufficient non-identical pairs for test",
                        }
                except Exception as e:
                    results[f"{model}_cot_vs_direct"] = {
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                        "error": str(e),
                    }

        return results

    def generate_visualizations(self, output_dir: str) -> None:
        """Generate visualization plots for the results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Basic accuracy comparison
        basic_metrics = self.calculate_basic_metrics()
        if not basic_metrics.empty:
            self._plot_accuracy_comparison(basic_metrics, output_path)

        # 2. Strategy comparison
        comparison = self.compare_strategies()
        if not comparison.empty:
            self._plot_strategy_comparison(comparison, output_path)

        # 3. Hallucination metrics (for all task types)
        hallucination_metrics = self.calculate_hallucination_metrics()
        if hallucination_metrics is not None and not hallucination_metrics.empty:
            self._plot_hallucination_metrics(hallucination_metrics, output_path)

    def _plot_accuracy_comparison(self, metrics_df: pd.DataFrame, output_path: Path) -> None:
        """Plot accuracy comparison across strategies."""
        plt.figure(figsize=(12, 8))

        # Create grouped bar plot
        sns.barplot(data=metrics_df, x="model", y="accuracy", hue="strategy")
        plt.title("Accuracy Comparison Across Models and Strategies")
        plt.ylabel("Accuracy")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(output_path / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_strategy_comparison(self, comparison_df: pd.DataFrame, output_path: Path) -> None:
        """Plot strategy improvement over direct approach."""
        plt.figure(figsize=(12, 8))

        # Melt the improvement columns for plotting
        improvement_cols = [col for col in comparison_df.columns if "_improvement" in col]

        if improvement_cols:
            melted = comparison_df.melt(
                id_vars=["model", "amount"],
                value_vars=improvement_cols,
                var_name="strategy",
                value_name="improvement",
            )

            # Clean strategy names
            melted["strategy"] = melted["strategy"].str.replace("_improvement", "")

            sns.barplot(data=melted, x="model", y="improvement", hue="strategy")
            plt.title("Strategy Improvement Over Direct Approach")
            plt.ylabel("Accuracy Improvement")
            plt.xlabel("Model")
            plt.xticks(rotation=45)
            plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            plt.savefig(output_path / "strategy_improvement.png", dpi=300, bbox_inches="tight")
            plt.close()

    def _plot_hallucination_metrics(self, metrics_df: pd.DataFrame, output_path: Path) -> None:
        """Plot hallucination-specific metrics."""
        # Set consistent figure size
        figsize = (12, 8)

        # Hallucination rate
        plt.figure(figsize=figsize)
        sns.barplot(data=metrics_df, x="model", y="hallucination_rate", hue="strategy")
        plt.title("Hallucination Rate by Model and Strategy")
        plt.ylabel("Hallucination Rate")
        plt.xticks(rotation=45)
        plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path / "hallucination_rate.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Miss rate
        plt.figure(figsize=figsize)
        sns.barplot(data=metrics_df, x="model", y="miss_rate", hue="strategy")
        plt.title("Miss Rate by Model and Strategy")
        plt.ylabel("Miss Rate")
        plt.xticks(rotation=45)
        plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path / "miss_rate.png", dpi=300, bbox_inches="tight")
        plt.close()

        # F1 Score
        plt.figure(figsize=figsize)
        sns.barplot(data=metrics_df, x="model", y="f1_score", hue="strategy")
        plt.title("F1 Score by Model and Strategy")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45)
        plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path / "f1_score.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Precision vs Recall scatter
        plt.figure(figsize=figsize)
        sns.scatterplot(
            data=metrics_df,
            x="precision",
            y="recall",
            hue="strategy",
            style="model",
            s=100,
        )
        plt.title("Precision vs Recall")
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path / "precision_recall.png", dpi=300, bbox_inches="tight")
        plt.close()

    def generate_summary_report(self, output_file: str) -> None:
        """Generate a comprehensive summary report."""
        with open(output_file, "w") as f:
            f.write("# Evaluation Results Summary Report\n\n")
            f.write(f"**Task Type:** {self.task_type}\n")
            f.write(f"**Number of Result Files:** {len(self.results)}\n\n")

            # Basic metrics
            f.write("## Basic Accuracy Metrics\n\n")
            basic_metrics = self.calculate_basic_metrics()
            if not basic_metrics.empty:
                f.write(basic_metrics.to_string(index=False))
                f.write("\n\n")

            # Strategy comparison
            f.write("## Strategy Comparison\n\n")
            comparison = self.compare_strategies()
            if not comparison.empty:
                f.write(comparison.to_string(index=False))
                f.write("\n\n")

            # Statistical significance
            f.write("## Statistical Significance Tests\n\n")
            significance = self.statistical_significance_test()
            for test_name, results in significance.items():
                f.write(f"**{test_name}:**\n")
                f.write(f"- p-value: {results['p_value']:.6f}\n")
                f.write(f"- Significant: {results['significant']}\n\n")

            # Hallucination metrics (for all task types)
            f.write("## Hallucination Metrics\n\n")
            hallucination_metrics = self.calculate_hallucination_metrics()
            if hallucination_metrics is not None and not hallucination_metrics.empty:
                f.write(hallucination_metrics.to_string(index=False))
                f.write("\n\n")

            # Key findings
            f.write("## Key Findings\n\n")
            self._write_key_findings(f, basic_metrics, comparison)

    def _write_key_findings(self, f, basic_metrics: pd.DataFrame, comparison: pd.DataFrame) -> None:
        """Write key findings based on the analysis."""
        if basic_metrics.empty:
            f.write("No sufficient data for key findings.\n")
            return

        # Best performing strategy per model
        f.write("### Best Performing Strategies:\n\n")
        best_strategies = basic_metrics.loc[basic_metrics.groupby("model")["accuracy"].idxmax()]
        for _, row in best_strategies.iterrows():
            f.write(f"- **{row['model']}**: {row['strategy']} ({row['accuracy']:.3f} accuracy)\n")
        f.write("\n")

        # Overall best performance
        overall_best = basic_metrics.loc[basic_metrics["accuracy"].idxmax()]
        f.write("### Overall Best Performance:\n")
        f.write(f"**{overall_best['model']}** with **{overall_best['strategy']}** strategy: ")
        f.write(f"{overall_best['accuracy']:.3f} accuracy\n\n")

        # Strategy improvements
        if not comparison.empty and "direct" in comparison.columns:
            f.write("### Strategy Improvements over Direct Approach:\n\n")
            for strategy in ["self_verification", "chain_of_thought"]:
                if f"{strategy}_improvement" in comparison.columns:
                    avg_improvement = comparison[f"{strategy}_improvement"].mean()
                    f.write(f"- **{strategy}**: {avg_improvement:+.3f} average improvement\n")
            f.write("\n")


@app.command()
def analyze(
    results_dir: str = typer.Option(
        ..., "--results-dir", "-r", help="Directory containing CSV result files"
    ),
    output_dir: str = typer.Option(
        "analysis_output", "--output-dir", "-o", help="Directory to save analysis outputs"
    ),
    visualizations: bool = typer.Option(
        False, "--visualizations", "-v", help="Generate visualization plots"
    ),
    report: bool = typer.Option(False, "--report", help="Generate summary report"),
) -> None:
    """Analyze evaluation results from visual puzzles and object detection scripts."""
    # Initialize analyzer
    analyzer = ResultsAnalyzer(results_dir)

    try:
        # Load results
        analyzer.load_results()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate basic metrics
        typer.echo("\n=== Basic Accuracy Metrics ===")
        basic_metrics = analyzer.calculate_basic_metrics()
        typer.echo(str(basic_metrics))

        # Generate strategy comparison
        typer.echo("\n=== Strategy Comparison ===")
        comparison = analyzer.compare_strategies()
        typer.echo(str(comparison))

        # Generate hallucination metrics if applicable
        if analyzer.task_type == "object_detection":
            typer.echo("\n=== Hallucination Metrics ===")
            hallucination_metrics = analyzer.calculate_hallucination_metrics()
            if hallucination_metrics is not None:
                typer.echo(str(hallucination_metrics))

        # Statistical significance tests
        typer.echo("\n=== Statistical Significance Tests ===")
        significance = analyzer.statistical_significance_test()
        for test_name, results in significance.items():
            typer.echo(
                f"{test_name}: p-value={results['p_value']:.6f}, "
                f"significant={results['significant']}"
            )

        # Generate visualizations
        if visualizations:
            typer.echo(f"\nGenerating visualizations in {output_dir}...")
            analyzer.generate_visualizations(output_dir)

        # Generate report
        if report:
            report_file = output_path / "summary_report.md"
            typer.echo(f"Generating summary report: {report_file}")
            analyzer.generate_summary_report(str(report_file))

        typer.echo("\nAnalysis complete!")

    except Exception as e:
        typer.echo(f"Error during analysis: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
