#!/usr/bin/env python3
"""
Prepare training dataset from decoded transactions.

This CLI script converts decoded transaction data (CSV or JSON) into
instruction-tuning format (JSONL) for HuggingFace Trainer.

Usage:
    python scripts/dataset/prepare_training_data.py \\
        --input data/processed/decoded_transactions.csv \\
        --output data/datasets \\
        --split 0.7 0.15 0.15

Output:
    - data/datasets/train.jsonl
    - data/datasets/validation.jsonl
    - data/datasets/test.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from eth_finetuning.dataset.preparation import prepare_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_decoded_transactions(input_path: Path) -> list[dict]:
    """
    Load decoded transactions from CSV or JSON file.

    Args:
        input_path: Path to input file

    Returns:
        List of decoded transaction dictionaries

    Raises:
        ValueError: If file format is unsupported
        FileNotFoundError: If input file doesn't exist
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        logger.info(f"Loading CSV file: {input_path}")
        df = pd.read_csv(input_path)

        # Convert DataFrame to list of dictionaries
        # Handle NaN values by converting to None
        transactions = df.to_dict(orient="records")

        # Convert pandas NaN to None
        for tx in transactions:
            for key, value in tx.items():
                if pd.isna(value):
                    tx[key] = None

        logger.info(f"Loaded {len(transactions)} transactions from CSV")
        return transactions

    elif suffix == ".json":
        logger.info(f"Loading JSON file: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            transactions = json.load(f)

        if not isinstance(transactions, list):
            raise ValueError("JSON file must contain a list of transactions")

        logger.info(f"Loaded {len(transactions)} transactions from JSON")
        return transactions

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. " f"Supported formats: .csv, .json"
        )


def parse_split_ratios(split_args: list[float]) -> tuple[float, float, float]:
    """
    Parse and validate split ratio arguments.

    Args:
        split_args: List of 3 floats for train/val/test ratios

    Returns:
        Tuple of (train, val, test) ratios

    Raises:
        ValueError: If ratios are invalid
    """
    if len(split_args) != 3:
        raise ValueError(
            f"Expected 3 split ratios (train val test), got {len(split_args)}"
        )

    train, val, test = split_args

    # Validate ratios are positive
    if any(r <= 0 for r in [train, val, test]):
        raise ValueError("Split ratios must be positive")

    # Validate ratios sum to 1.0 (with small tolerance)
    total = train + val + test
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0 (got {total}). "
            f"Use ratios like: 0.7 0.15 0.15"
        )

    return train, val, test


def main():
    """Main entry point for dataset preparation CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare training dataset from decoded transactions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset with default 70/15/15 split
  python scripts/dataset/prepare_training_data.py \\
      --input data/processed/decoded_transactions.csv \\
      --output data/datasets

  # Custom split ratios
  python scripts/dataset/prepare_training_data.py \\
      --input data/raw/transactions.json \\
      --output data/datasets \\
      --split 0.8 0.1 0.1

  # Disable stratification (random split)
  python scripts/dataset/prepare_training_data.py \\
      --input data/processed/decoded_transactions.csv \\
      --output data/datasets \\
      --no-stratify
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input file (CSV or JSON with decoded transactions)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save train.jsonl, validation.jsonl, test.jsonl",
    )

    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios for train/val/test (must sum to 1.0). Default: 0.7 0.15 0.15",
    )

    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratification by protocol (use random split instead)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    try:
        # Parse and validate split ratios
        split_ratios = parse_split_ratios(args.split)
        logger.info(
            f"Using split ratios: train={split_ratios[0]:.2%}, "
            f"val={split_ratios[1]:.2%}, test={split_ratios[2]:.2%}"
        )

        # Load decoded transactions
        decoded_txs = load_decoded_transactions(args.input)

        if not decoded_txs:
            logger.error("No transactions found in input file")
            return 1

        # Prepare dataset
        stratify = not args.no_stratify
        logger.info(f"Preparing dataset with stratification: {stratify}")

        counts = prepare_dataset(
            decoded_txs=decoded_txs,
            output_dir=args.output,
            split_ratios=split_ratios,
            stratify_by_protocol=stratify,
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("Dataset preparation complete!")
        logger.info(f"  Train:      {counts['train']} examples")
        logger.info(f"  Validation: {counts['validation']} examples")
        logger.info(f"  Test:       {counts['test']} examples")
        logger.info(f"  Total:      {sum(counts.values())} examples")
        logger.info("=" * 60)
        logger.info(f"Output directory: {args.output.resolve()}")
        logger.info("  - train.jsonl")
        logger.info("  - validation.jsonl")
        logger.info("  - test.jsonl")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
