#!/usr/bin/env python3
"""
CLI script for evaluating fine-tuned Ethereum intent extraction models.

This script provides a command-line interface for running comprehensive
evaluations of fine-tuned models. It loads the model, runs inference on
a test dataset, calculates accuracy metrics, and generates reports.

Usage:
    # Basic evaluation
    python scripts/evaluation/evaluate_model.py \
        --model models/fine-tuned/eth-intent-extractor \
        --test-data data/datasets/test.jsonl \
        --output outputs/metrics/results.json

    # Evaluation with custom settings
    python scripts/evaluation/evaluate_model.py \
        --model models/fine-tuned/eth-intent-extractor \
        --test-data data/datasets/test.jsonl \
        --output outputs/metrics/results.json \
        --max-tokens 512 \
        --temperature 0.1 \
        --tolerance 0.01 \
        --no-report

Requirements:
    - Fine-tuned model with adapter files
    - Test dataset in JSONL format (instruction/input/output)
    - CUDA-capable GPU (recommended) or CPU

Expected Output:
    - results.json: Comprehensive metrics JSON
    - results_predictions.json: Raw predictions for debugging
    - results_report.md: Human-readable markdown report
"""

import logging
import sys
from pathlib import Path

import click

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eth_finetuning.evaluation import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to fine-tuned model directory (with adapter files)",
)
@click.option(
    "--test-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to test dataset JSONL file",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Path to save evaluation results JSON",
)
@click.option(
    "--max-tokens",
    type=int,
    default=512,
    help="Maximum tokens to generate per prediction (default: 512)",
)
@click.option(
    "--temperature",
    type=float,
    default=0.1,
    help="Sampling temperature for generation (default: 0.1 for deterministic)",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.01,
    help="Amount accuracy tolerance as decimal (default: 0.01 = 1%%)",
)
@click.option(
    "--no-report",
    is_flag=True,
    default=False,
    help="Skip generating markdown report",
)
def main(
    model: str,
    test_data: str,
    output: str,
    max_tokens: int,
    temperature: float,
    tolerance: float,
    no_report: bool,
):
    """
    Evaluate fine-tuned model on Ethereum transaction intent extraction.

    This script runs comprehensive evaluation including:
    - Batch inference on test dataset
    - Accuracy calculations (amounts, addresses, protocols)
    - Confusion matrix generation
    - Per-protocol performance breakdown
    - Markdown report generation

    The evaluation metrics quantify model performance against success targets:
    - ≥90% accuracy on amounts, addresses, and protocols
    - ≥60 Flesch Reading Ease score (if applicable)
    - Zero failed JSON parses

    Example:
        python scripts/evaluation/evaluate_model.py \\
            --model models/fine-tuned/eth-intent-extractor \\
            --test-data data/datasets/test.jsonl \\
            --output outputs/metrics/results.json
    """
    logger.info("=" * 80)
    logger.info("Ethereum Intent Extraction - Model Evaluation")
    logger.info("=" * 80)

    # Validate paths
    model_path = Path(model)
    test_data_path = Path(test_data)
    output_path = Path(output)

    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)

    if not test_data_path.exists():
        logger.error(f"Test data not found: {test_data_path}")
        sys.exit(1)

    # Check for adapter files
    adapter_config = model_path / "adapter_config.json"
    if not adapter_config.exists():
        logger.warning(
            f"adapter_config.json not found in {model_path}. "
            "Ensure this is a fine-tuned model directory."
        )

    logger.info(f"Model: {model_path}")
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Max tokens: {max_tokens}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Tolerance: {tolerance:.2%}")
    logger.info(f"Generate report: {not no_report}")

    # Run evaluation
    try:
        metrics = evaluate_model(
            model_path=model_path,
            test_data_path=test_data_path,
            output_path=output_path,
            max_new_tokens=max_tokens,
            temperature=temperature,
            tolerance=tolerance,
            generate_report=not no_report,
        )

        logger.info("=" * 80)
        logger.info("Evaluation Summary")
        logger.info("=" * 80)
        logger.info(f"✓ Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        logger.info(f"✓ Amount Accuracy: {metrics['amount_accuracy']:.2%}")
        logger.info(f"✓ Address Accuracy: {metrics['address_accuracy']:.2%}")
        logger.info(f"✓ Protocol Accuracy: {metrics['protocol_accuracy']:.2%}")

        if metrics.get("flesch_score", 0) > 0:
            logger.info(f"✓ Flesch Reading Ease: {metrics['flesch_score']:.1f}")

        logger.info(f"✓ Total Samples: {metrics['total_samples']}")
        logger.info(f"✓ Failed Parses: {metrics.get('failed_parses', 0)}")

        # Success check
        overall_acc = metrics["overall_accuracy"]
        if overall_acc >= 0.90:
            logger.info("=" * 80)
            logger.info("🎉 SUCCESS: Model meets 90% accuracy target!")
            logger.info("=" * 80)
        elif overall_acc >= 0.85:
            logger.info("=" * 80)
            logger.info("⚠️  CLOSE: Model near target, minor improvements needed")
            logger.info("=" * 80)
        else:
            logger.info("=" * 80)
            logger.info("❌ NEEDS WORK: Model below target, improvements required")
            logger.info("=" * 80)

        logger.info(f"\nResults saved to: {output_path}")

        if not no_report:
            report_path = output_path.parent / f"{output_path.stem}_report.md"
            logger.info(f"Report saved to: {report_path}")

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

    logger.info("\nNext steps:")
    logger.info("1. Review the evaluation report for detailed insights")
    logger.info("2. Check per-protocol metrics to identify specific issues")
    logger.info("3. If accuracy is low, consider:")
    logger.info("   - Increasing training epochs or dataset size")
    logger.info("   - Adjusting hyperparameters (learning rate, LoRA rank)")
    logger.info("   - Reviewing and improving training data quality")


if __name__ == "__main__":
    main()
