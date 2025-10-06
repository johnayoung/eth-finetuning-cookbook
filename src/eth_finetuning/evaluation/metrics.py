"""
Metrics calculation for Ethereum intent extraction evaluation.

This module provides functions for calculating various accuracy metrics
including amount accuracy (with tolerance), address accuracy (checksummed),
protocol accuracy, and Flesch Reading Ease scores.

Usage:
    from eth_finetuning.evaluation.metrics import calculate_accuracy_metrics

    metrics = calculate_accuracy_metrics(predictions, ground_truth)
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
"""

import json
import logging
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def calculate_amount_accuracy(
    predicted_amounts: Sequence[float | str],
    ground_truth_amounts: Sequence[float | str],
    tolerance: float = 0.01,
) -> float:
    """
    Calculate amount accuracy with tolerance for floating point errors.

    Compares predicted and ground truth amounts with ±1% tolerance by default.
    Handles both numeric and string inputs (converts strings to float).

    Args:
        predicted_amounts: List of predicted amounts (numbers or strings)
        ground_truth_amounts: List of ground truth amounts (numbers or strings)
        tolerance: Relative tolerance for match (default: 0.01 = 1%)

    Returns:
        Accuracy as float between 0 and 1

    Raises:
        ValueError: If lists have different lengths

    Notes:
        - Matches are considered correct if |predicted - actual| <= tolerance * actual
        - Empty amounts or None values are treated as mismatches
        - Non-numeric strings are treated as mismatches
    """
    if len(predicted_amounts) != len(ground_truth_amounts):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted_amounts)}, "
            f"ground_truth={len(ground_truth_amounts)}"
        )

    if not predicted_amounts:
        return 0.0

    correct = 0
    total = len(predicted_amounts)

    for pred, truth in zip(predicted_amounts, ground_truth_amounts):
        try:
            # Convert to float if string
            pred_val = float(pred) if pred is not None else None
            truth_val = float(truth) if truth is not None else None

            if pred_val is None or truth_val is None:
                continue  # Mismatch

            # Check if within tolerance
            if truth_val == 0:
                # Exact match required for zero
                if pred_val == 0:
                    correct += 1
            else:
                # Relative tolerance check
                relative_error = abs(pred_val - truth_val) / abs(truth_val)
                if relative_error <= tolerance:
                    correct += 1

        except (ValueError, TypeError):
            # Non-numeric values are mismatches
            continue

    return correct / total if total > 0 else 0.0


def calculate_address_accuracy(
    predicted_addresses: Sequence[str],
    ground_truth_addresses: Sequence[str],
    case_sensitive: bool = False,
) -> float:
    """
    Calculate address accuracy with exact string matching.

    Compares Ethereum addresses after normalizing to checksummed format.
    By default, comparison is case-insensitive after checksumming.

    Args:
        predicted_addresses: List of predicted Ethereum addresses
        ground_truth_addresses: List of ground truth Ethereum addresses
        case_sensitive: Whether to enforce case-sensitive matching (default: False)

    Returns:
        Accuracy as float between 0 and 1

    Raises:
        ValueError: If lists have different lengths

    Notes:
        - Addresses are normalized to lowercase before comparison (unless case_sensitive)
        - Empty or None addresses are treated as mismatches
        - Addresses must start with '0x' to be valid
    """
    if len(predicted_addresses) != len(ground_truth_addresses):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted_addresses)}, "
            f"ground_truth={len(ground_truth_addresses)}"
        )

    if not predicted_addresses:
        return 0.0

    correct = 0
    total = len(predicted_addresses)

    for pred, truth in zip(predicted_addresses, ground_truth_addresses):
        try:
            # Normalize addresses
            pred_norm = _normalize_address(pred, case_sensitive)
            truth_norm = _normalize_address(truth, case_sensitive)

            if pred_norm and truth_norm and pred_norm == truth_norm:
                correct += 1

        except Exception:
            # Malformed addresses are mismatches
            continue

    return correct / total if total > 0 else 0.0


def calculate_protocol_accuracy(
    predicted_protocols: Sequence[str],
    ground_truth_protocols: Sequence[str],
) -> float:
    """
    Calculate protocol classification accuracy.

    Simple exact match accuracy for protocol identification.
    Case-insensitive comparison after normalization.

    Args:
        predicted_protocols: List of predicted protocols
        ground_truth_protocols: List of ground truth protocols

    Returns:
        Accuracy as float between 0 and 1

    Raises:
        ValueError: If lists have different lengths

    Notes:
        - Protocols are normalized to lowercase
        - Empty or None protocols are treated as mismatches
        - Common protocol names: "ethereum", "erc20", "uniswap_v2", "uniswap_v3"
    """
    if len(predicted_protocols) != len(ground_truth_protocols):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted_protocols)}, "
            f"ground_truth={len(ground_truth_protocols)}"
        )

    if not predicted_protocols:
        return 0.0

    correct = 0
    total = len(predicted_protocols)

    for pred, truth in zip(predicted_protocols, ground_truth_protocols):
        try:
            # Normalize to lowercase
            pred_norm = pred.lower().strip() if pred else ""
            truth_norm = truth.lower().strip() if truth else ""

            if pred_norm and truth_norm and pred_norm == truth_norm:
                correct += 1

        except Exception:
            # Malformed protocols are mismatches
            continue

    return correct / total if total > 0 else 0.0


def calculate_flesch_score(texts: list[str]) -> float:
    """
    Calculate average Flesch Reading Ease score for generated texts.

    Higher scores indicate easier readability. Target: 60+ for accessible text.
    Score ranges:
        - 90-100: Very easy (5th grade)
        - 60-70: Standard (8th-9th grade)
        - 30-50: Difficult (college)
        - 0-30: Very difficult (college graduate)

    Args:
        texts: List of text strings to evaluate

    Returns:
        Average Flesch Reading Ease score (0-100)

    Notes:
        - Requires textstat library
        - Empty strings are skipped
        - Returns 0.0 if no valid texts or textstat unavailable
        - Only applicable if model generates natural language descriptions
    """
    try:
        import textstat  # type: ignore
    except ImportError:
        logger.warning("textstat library not installed, cannot calculate Flesch score")
        return 0.0

    if not texts:
        return 0.0

    scores = []
    for text in texts:
        if text and isinstance(text, str) and len(text.strip()) > 0:
            try:
                score = textstat.flesch_reading_ease(text)  # type: ignore
                scores.append(score)
            except Exception as e:
                logger.debug(f"Failed to calculate Flesch score for text: {e}")
                continue

    return float(np.mean(scores)) if scores else 0.0


def create_confusion_matrix(
    predicted_protocols: list[str],
    ground_truth_protocols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Create confusion matrix for protocol classification.

    Args:
        predicted_protocols: List of predicted protocols
        ground_truth_protocols: List of ground truth protocols

    Returns:
        Tuple of (confusion_matrix, labels)
        - confusion_matrix: NxN numpy array where N is number of unique protocols
        - labels: List of protocol names corresponding to matrix rows/columns

    Raises:
        ValueError: If lists have different lengths

    Notes:
        - Protocols are normalized to lowercase
        - Matrix[i,j] represents count of truth=labels[i], predicted=labels[j]
    """
    if len(predicted_protocols) != len(ground_truth_protocols):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted_protocols)}, "
            f"ground_truth={len(ground_truth_protocols)}"
        )

    # Normalize protocols
    pred_norm = [p.lower().strip() if p else "unknown" for p in predicted_protocols]
    truth_norm = [p.lower().strip() if p else "unknown" for p in ground_truth_protocols]

    # Get unique labels (sorted for consistency)
    labels = sorted(set(truth_norm) | set(pred_norm))

    # Create confusion matrix
    cm = confusion_matrix(truth_norm, pred_norm, labels=labels)

    return cm, labels


def calculate_accuracy_metrics(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """
    Calculate comprehensive accuracy metrics from predictions and ground truth.

    Extracts relevant fields from prediction/ground truth dictionaries and
    calculates accuracy metrics for amounts, addresses, protocols, and
    overall accuracy.

    Args:
        predictions: List of predicted intent dictionaries
        ground_truth: List of ground truth intent dictionaries
        tolerance: Tolerance for amount matching (default: 0.01 = 1%)

    Returns:
        Dictionary with metrics:
        {
            "overall_accuracy": float,
            "amount_accuracy": float,
            "address_accuracy": float,
            "protocol_accuracy": float,
            "flesch_score": float (if text descriptions present),
            "total_samples": int,
            "confusion_matrix": list[list[int]],
            "confusion_matrix_labels": list[str]
        }

    Raises:
        ValueError: If predictions and ground_truth have different lengths

    Notes:
        - Expects dictionaries with keys: "amounts", "from"/"to", "protocol"
        - Handles missing fields gracefully (treated as mismatches)
        - Flesch score only calculated if "description" field present
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"ground_truth={len(ground_truth)}"
        )

    if not predictions:
        return {
            "overall_accuracy": 0.0,
            "amount_accuracy": 0.0,
            "address_accuracy": 0.0,
            "protocol_accuracy": 0.0,
            "flesch_score": 0.0,
            "total_samples": 0,
        }

    # Extract fields for each metric
    pred_amounts = _extract_amounts(predictions)
    truth_amounts = _extract_amounts(ground_truth)

    pred_addresses = _extract_addresses(predictions)
    truth_addresses = _extract_addresses(ground_truth)

    pred_protocols = _extract_protocols(predictions)
    truth_protocols = _extract_protocols(ground_truth)

    # Calculate individual metrics
    amount_acc = calculate_amount_accuracy(pred_amounts, truth_amounts, tolerance)
    address_acc = calculate_address_accuracy(pred_addresses, truth_addresses)
    protocol_acc = calculate_protocol_accuracy(pred_protocols, truth_protocols)

    # Calculate overall accuracy (average of individual metrics)
    overall_acc = (amount_acc + address_acc + protocol_acc) / 3

    # Calculate Flesch score if descriptions present
    descriptions = _extract_descriptions(predictions)
    flesch = calculate_flesch_score(descriptions) if descriptions else 0.0

    # Create confusion matrix
    cm, labels = create_confusion_matrix(pred_protocols, truth_protocols)

    return {
        "overall_accuracy": overall_acc,
        "amount_accuracy": amount_acc,
        "address_accuracy": address_acc,
        "protocol_accuracy": protocol_acc,
        "flesch_score": flesch,
        "total_samples": len(predictions),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
    }


def calculate_per_protocol_metrics(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    tolerance: float = 0.01,
) -> dict[str, dict[str, float]]:
    """
    Calculate accuracy metrics broken down by protocol.

    Args:
        predictions: List of predicted intent dictionaries
        ground_truth: List of ground truth intent dictionaries
        tolerance: Tolerance for amount matching (default: 0.01 = 1%)

    Returns:
        Dictionary mapping protocol names to their metrics:
        {
            "ethereum": {"amount_acc": 0.95, "address_acc": 0.98, ...},
            "uniswap_v2": {"amount_acc": 0.92, "address_acc": 0.97, ...},
            ...
        }

    Notes:
        - Groups samples by ground truth protocol
        - Calculates metrics independently for each protocol
        - Useful for identifying protocol-specific performance issues
    """
    if not predictions or not ground_truth:
        return {}

    # Group samples by protocol
    protocol_samples: dict[str, list[tuple[dict, dict]]] = {}

    for pred, truth in zip(predictions, ground_truth):
        protocol = _extract_protocol(truth)
        if protocol:
            if protocol not in protocol_samples:
                protocol_samples[protocol] = []
            protocol_samples[protocol].append((pred, truth))

    # Calculate metrics for each protocol
    per_protocol = {}

    for protocol, samples in protocol_samples.items():
        pred_list = [s[0] for s in samples]
        truth_list = [s[1] for s in samples]

        metrics = calculate_accuracy_metrics(pred_list, truth_list, tolerance)

        per_protocol[protocol] = {
            "amount_accuracy": metrics["amount_accuracy"],
            "address_accuracy": metrics["address_accuracy"],
            "protocol_accuracy": metrics["protocol_accuracy"],
            "overall_accuracy": metrics["overall_accuracy"],
            "total_samples": len(samples),
        }

    return per_protocol


# Helper functions for extracting fields from intent dictionaries


def _extract_amounts(intents: list[dict[str, Any]]) -> list[float]:
    """Extract first amount from each intent."""
    amounts = []
    for intent in intents:
        if "amounts" in intent and intent["amounts"]:
            # Get first amount
            amount = (
                intent["amounts"][0]
                if isinstance(intent["amounts"], list)
                else intent["amounts"]
            )
            amounts.append(amount)
        elif "amount" in intent:
            amounts.append(intent["amount"])
        else:
            amounts.append(0.0)
    return amounts


def _extract_addresses(intents: list[dict[str, Any]]) -> list[str]:
    """Extract primary address from each intent (from/to field)."""
    addresses = []
    for intent in intents:
        # Try 'to' field first, then 'from'
        address = intent.get("to") or intent.get("from") or ""
        addresses.append(str(address) if address else "")
    return addresses


def _extract_protocols(intents: list[dict[str, Any]]) -> list[str]:
    """Extract protocol from each intent."""
    return [str(intent.get("protocol", "")).lower() for intent in intents]


def _extract_protocol(intent: dict[str, Any]) -> str:
    """Extract protocol from single intent."""
    return str(intent.get("protocol", "")).lower()


def _extract_descriptions(intents: list[dict[str, Any]]) -> list[str]:
    """Extract text descriptions from intents (if present)."""
    descriptions = []
    for intent in intents:
        if "description" in intent and intent["description"]:
            descriptions.append(str(intent["description"]))
    return descriptions


def _normalize_address(address: str | None, case_sensitive: bool = False) -> str:
    """Normalize Ethereum address for comparison."""
    if not address:
        return ""

    address_str = str(address).strip()

    # Must start with 0x
    if not address_str.startswith("0x"):
        return ""

    # Return lowercase unless case sensitive
    return address_str if case_sensitive else address_str.lower()
