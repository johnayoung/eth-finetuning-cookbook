"""
Dataset preparation pipeline for fine-tuning.

This module provides utilities for preparing training datasets from decoded
transactions, including train/val/test splitting, data validation, and
JSONL export for HuggingFace Trainer.

Usage:
    from eth_finetuning.dataset.preparation import prepare_dataset

    prepare_dataset(
        decoded_txs=transactions,
        output_dir="data/datasets",
        split_ratios=(0.7, 0.15, 0.15)
    )
"""

import json
import logging
from pathlib import Path
from typing import Any

from web3 import Web3

from .intent_extraction import extract_intents_batch
from .templates import format_training_examples_batch

logger = logging.getLogger(__name__)


def prepare_dataset(
    decoded_txs: list[dict[str, Any]],
    output_dir: str | Path,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    stratify_by_protocol: bool = True,
) -> dict[str, int]:
    """
    Prepare training dataset from decoded transactions.

    Converts decoded transactions to instruction-tuning format, splits into
    train/val/test sets, validates data quality, and exports to JSONL files.

    Args:
        decoded_txs: List of decoded transaction dictionaries
        output_dir: Directory to save train.jsonl, validation.jsonl, test.jsonl
        split_ratios: Tuple of (train, val, test) ratios (must sum to 1.0)
        stratify_by_protocol: If True, maintain protocol distribution across splits

    Returns:
        Dictionary with counts: {"train": N, "validation": M, "test": K}

    Raises:
        ValueError: If split_ratios don't sum to 1.0 or decoded_txs is empty
        ValueError: If data validation fails

    Notes:
        - Creates output_dir if it doesn't exist
        - Overwrites existing JSONL files
        - Validates all data before splitting
        - Stratification ensures balanced protocol representation
    """
    if not decoded_txs:
        raise ValueError("decoded_txs cannot be empty")

    # Validate split ratios
    if not abs(sum(split_ratios) - 1.0) < 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0 (got {sum(split_ratios)})")

    train_ratio, val_ratio, test_ratio = split_ratios

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing dataset from {len(decoded_txs)} transactions")

    # Step 1: Extract intents
    logger.info("Extracting intents from transactions...")
    intents = extract_intents_batch(decoded_txs)

    if not intents:
        raise ValueError("No intents extracted from decoded transactions")

    # Filter decoded_txs to match extracted intents (some may have failed)
    tx_hashes = {intent["tx_hash"] for intent in intents}
    decoded_txs_filtered = [tx for tx in decoded_txs if tx.get("tx_hash") in tx_hashes]

    logger.info(f"Successfully extracted {len(intents)} intents")

    # Step 2: Validate data quality
    logger.info("Validating data quality...")
    validate_data(decoded_txs_filtered, intents)
    logger.info("Data validation passed")

    # Step 3: Format training examples
    logger.info("Formatting training examples...")
    examples = format_training_examples_batch(decoded_txs_filtered, intents)

    if not examples:
        raise ValueError("No training examples formatted")

    logger.info(f"Formatted {len(examples)} training examples")

    # Step 4: Split dataset
    logger.info("Splitting dataset...")
    if stratify_by_protocol:
        train_examples, val_examples, test_examples = _stratified_split(
            examples, decoded_txs_filtered, split_ratios
        )
    else:
        train_examples, val_examples, test_examples = _random_split(
            examples, split_ratios
        )

    logger.info(
        f"Split complete: train={len(train_examples)}, "
        f"val={len(val_examples)}, test={len(test_examples)}"
    )

    # Step 5: Export to JSONL
    logger.info("Exporting datasets to JSONL...")
    _export_jsonl(train_examples, output_dir / "train.jsonl")
    _export_jsonl(val_examples, output_dir / "validation.jsonl")
    _export_jsonl(test_examples, output_dir / "test.jsonl")

    logger.info(f"Dataset preparation complete. Files saved to {output_dir}")

    return {
        "train": len(train_examples),
        "validation": len(val_examples),
        "test": len(test_examples),
    }


def validate_data(
    decoded_txs: list[dict[str, Any]], intents: list[dict[str, Any]]
) -> None:
    """
    Validate data quality before dataset preparation.

    Checks for:
    - No null values in critical fields
    - Addresses are properly checksummed
    - Amounts are numeric
    - Protocols are valid
    - Intents match decoded transactions

    Args:
        decoded_txs: List of decoded transaction dictionaries
        intents: List of intent dictionaries

    Raises:
        ValueError: If validation fails with detailed error message

    Notes:
        - Critical fields: action, protocol, status
        - Addresses must pass Web3.is_checksum_address()
        - Amounts can be 0 but must be numeric
    """
    if len(decoded_txs) != len(intents):
        raise ValueError(
            f"Mismatch: {len(decoded_txs)} transactions but {len(intents)} intents"
        )

    errors = []

    for i, (decoded_tx, intent) in enumerate(zip(decoded_txs, intents)):
        tx_hash = decoded_tx.get("tx_hash", f"index_{i}")

        # Validate critical fields are not None
        for field in ["action", "protocol", "status"]:
            if decoded_tx.get(field) is None:
                errors.append(f"{tx_hash}: missing '{field}' in decoded_tx")

        for field in ["action", "protocol", "outcome"]:
            if intent.get(field) is None:
                errors.append(f"{tx_hash}: missing '{field}' in intent")

        # Validate protocol consistency
        if decoded_tx.get("protocol") != intent.get("protocol"):
            errors.append(
                f"{tx_hash}: protocol mismatch "
                f"({decoded_tx.get('protocol')} vs {intent.get('protocol')})"
            )

        # Validate addresses are checksummed
        _validate_addresses(decoded_tx, tx_hash, errors)

        # Validate amounts are numeric
        _validate_amounts(decoded_tx, intent, tx_hash, errors)

    if errors:
        error_summary = "\n".join(errors[:10])  # Show first 10 errors
        if len(errors) > 10:
            error_summary += f"\n... and {len(errors) - 10} more errors"
        raise ValueError(f"Data validation failed:\n{error_summary}")


def _validate_addresses(
    decoded_tx: dict[str, Any], tx_hash: str, errors: list[str]
) -> None:
    """Validate addresses are checksummed."""
    address_fields = []

    protocol = decoded_tx.get("protocol")
    if protocol == "ethereum":
        address_fields = ["from", "to"]
    elif protocol == "erc20":
        address_fields = ["from", "to", "token_address"]
    elif protocol in ["uniswap_v2", "uniswap_v3"]:
        address_fields = ["pool_address", "token_in", "token_out"]

    for field in address_fields:
        address = decoded_tx.get(field)
        if address and not Web3.is_checksum_address(address):
            errors.append(f"{tx_hash}: '{field}' is not checksummed: {address}")


def _validate_amounts(
    decoded_tx: dict[str, Any], intent: dict[str, Any], tx_hash: str, errors: list[str]
) -> None:
    """Validate amounts are numeric."""
    protocol = decoded_tx.get("protocol")

    # Check decoded_tx amounts
    amount_fields = []
    if protocol == "ethereum":
        amount_fields = ["amount_wei", "amount_eth"]
    elif protocol == "erc20":
        amount_fields = ["amount"]
        if decoded_tx.get("amount_formatted") is not None:
            amount_fields.append("amount_formatted")
    elif protocol in ["uniswap_v2", "uniswap_v3"]:
        amount_fields = ["amount_in", "amount_out"]
        if decoded_tx.get("amount_in_formatted") is not None:
            amount_fields.append("amount_in_formatted")
        if decoded_tx.get("amount_out_formatted") is not None:
            amount_fields.append("amount_out_formatted")

    for field in amount_fields:
        amount = decoded_tx.get(field)
        if amount is not None and not isinstance(amount, (int, float)):
            errors.append(f"{tx_hash}: '{field}' is not numeric: {type(amount)}")

    # Check intent amounts
    amounts = intent.get("amounts", [])
    if not isinstance(amounts, list):
        errors.append(f"{tx_hash}: intent 'amounts' is not a list: {type(amounts)}")
    else:
        for idx, amount in enumerate(amounts):
            if not isinstance(amount, (int, float)):
                errors.append(
                    f"{tx_hash}: intent amounts[{idx}] is not numeric: {type(amount)}"
                )


def _stratified_split(
    examples: list[dict[str, Any]],
    decoded_txs: list[dict[str, Any]],
    split_ratios: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split dataset with stratification by protocol.

    Ensures protocol distribution is preserved across train/val/test splits.

    Args:
        examples: Training examples
        decoded_txs: Decoded transactions (for protocol info)
        split_ratios: (train, val, test) ratios

    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    train_ratio, val_ratio, test_ratio = split_ratios

    # Group by protocol
    protocol_groups: dict[str, list[tuple[dict, dict]]] = {}
    for example, decoded_tx in zip(examples, decoded_txs):
        protocol = decoded_tx.get("protocol", "unknown")
        if protocol not in protocol_groups:
            protocol_groups[protocol] = []
        protocol_groups[protocol].append((example, decoded_tx))

    train_examples = []
    val_examples = []
    test_examples = []

    # Split each protocol group
    for protocol, group in protocol_groups.items():
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Extract just the examples (not decoded_txs)
        group_examples = [ex for ex, _ in group]

        train_examples.extend(group_examples[:train_end])
        val_examples.extend(group_examples[train_end:val_end])
        test_examples.extend(group_examples[val_end:])

        logger.debug(
            f"Protocol {protocol}: {n} total -> "
            f"train={train_end}, val={val_end - train_end}, test={n - val_end}"
        )

    return train_examples, val_examples, test_examples


def _random_split(
    examples: list[dict[str, Any]], split_ratios: tuple[float, float, float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split dataset randomly without stratification.

    Args:
        examples: Training examples
        split_ratios: (train, val, test) ratios

    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    train_ratio, val_ratio, test_ratio = split_ratios

    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_examples = examples[:train_end]
    val_examples = examples[train_end:val_end]
    test_examples = examples[val_end:]

    return train_examples, val_examples, test_examples


def _export_jsonl(examples: list[dict[str, Any]], output_path: Path) -> None:
    """
    Export training examples to JSONL file.

    Each line is a valid JSON object with instruction, input, output fields.

    Args:
        examples: Training examples
        output_path: Path to output JSONL file

    Raises:
        IOError: If file cannot be written
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + "\n")

    logger.info(f"Exported {len(examples)} examples to {output_path}")
