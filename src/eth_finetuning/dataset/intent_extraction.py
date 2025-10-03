"""
Convert decoded transactions to structured intent JSON for fine-tuning.

This module transforms decoded transaction data (from ETH, ERC-20, and Uniswap decoders)
into structured intent JSON suitable for instruction-tuning language models.

Intent Format:
    {
        "action": "transfer" | "swap",
        "assets": [token_symbols],
        "protocol": "ethereum" | "erc20" | "uniswap_v2" | "uniswap_v3",
        "outcome": "success" | "failed",
        "amounts": [numeric_values]
    }

Usage:
    from eth_finetuning.dataset.intent_extraction import extract_intent

    intent = extract_intent(decoded_transaction)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_intent(decoded_tx: dict[str, Any]) -> dict[str, Any]:
    """
    Extract structured intent from a decoded transaction.

    Converts protocol-specific decoded transaction data into a unified intent format
    suitable for training language models on blockchain transaction understanding.

    Args:
        decoded_tx: Decoded transaction dictionary from decoders
            (eth.py, erc20.py, uniswap/v2.py, uniswap/v3.py)

    Returns:
        Intent dictionary with structure:
        {
            "action": str - Type of action ("transfer", "swap")
            "assets": list[str] - Token symbols involved
            "protocol": str - Protocol name
            "outcome": str - Transaction status ("success" or "failed")
            "amounts": list[float] - Numeric amounts (formatted with decimals)
        }

    Raises:
        ValueError: If decoded_tx is missing required fields or has invalid protocol

    Notes:
        - Handles all transaction types: ETH, ERC-20, Uniswap V2/V3
        - Uses formatted amounts when available, falls back to raw amounts
        - Token symbols default to "UNKNOWN" if not available
        - Preserves transaction outcome (success/failed) for learning
    """
    if not decoded_tx:
        raise ValueError("decoded_tx cannot be empty")

    protocol = decoded_tx.get("protocol")
    if not protocol:
        raise ValueError("decoded_tx must have 'protocol' field")

    action = decoded_tx.get("action")
    if not action:
        raise ValueError("decoded_tx must have 'action' field")

    # Extract status/outcome
    status = decoded_tx.get("status", "success")
    outcome = "success" if status == "success" else "failed"

    # Extract intent based on protocol
    if protocol == "ethereum":
        return _extract_eth_intent(decoded_tx, action, outcome)
    elif protocol == "erc20":
        return _extract_erc20_intent(decoded_tx, action, outcome)
    elif protocol in ["uniswap_v2", "uniswap_v3"]:
        return _extract_uniswap_intent(decoded_tx, action, outcome, protocol)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def _extract_eth_intent(
    decoded_tx: dict[str, Any], action: str, outcome: str
) -> dict[str, Any]:
    """
    Extract intent from ETH transfer.

    Args:
        decoded_tx: Decoded ETH transfer from eth.py
        action: Transaction action type
        outcome: Transaction outcome

    Returns:
        Intent dictionary
    """
    # Get amount in ETH (formatted)
    amount_eth = decoded_tx.get("amount_eth", 0.0)
    if not isinstance(amount_eth, (int, float)):
        logger.warning(
            f"Invalid amount_eth type: {type(amount_eth)}, defaulting to 0.0"
        )
        amount_eth = 0.0

    return {
        "action": action,
        "assets": ["ETH"],
        "protocol": "ethereum",
        "outcome": outcome,
        "amounts": [float(amount_eth)],
    }


def _extract_erc20_intent(
    decoded_tx: dict[str, Any], action: str, outcome: str
) -> dict[str, Any]:
    """
    Extract intent from ERC-20 transfer.

    Args:
        decoded_tx: Decoded ERC-20 transfer from erc20.py
        action: Transaction action type
        outcome: Transaction outcome

    Returns:
        Intent dictionary
    """
    # Get token symbol (default to UNKNOWN if not available)
    token_symbol = decoded_tx.get("token_symbol", "UNKNOWN")

    # Get formatted amount (with decimals), fall back to raw amount
    amount_formatted = decoded_tx.get("amount_formatted")
    if amount_formatted is not None and isinstance(amount_formatted, (int, float)):
        amount = float(amount_formatted)
    else:
        # Fall back to raw amount
        raw_amount = decoded_tx.get("amount", 0)
        amount = float(raw_amount) if isinstance(raw_amount, (int, float)) else 0.0

    return {
        "action": action,
        "assets": [token_symbol],
        "protocol": "erc20",
        "outcome": outcome,
        "amounts": [amount],
    }


def _extract_uniswap_intent(
    decoded_tx: dict[str, Any], action: str, outcome: str, protocol: str
) -> dict[str, Any]:
    """
    Extract intent from Uniswap swap (V2 or V3).

    Args:
        decoded_tx: Decoded Uniswap swap from uniswap/v2.py or uniswap/v3.py
        action: Transaction action type
        outcome: Transaction outcome
        protocol: Protocol name ("uniswap_v2" or "uniswap_v3")

    Returns:
        Intent dictionary
    """
    # Get token symbols (default to UNKNOWN if not available)
    token_in_symbol = decoded_tx.get("token_in_symbol", "UNKNOWN")
    token_out_symbol = decoded_tx.get("token_out_symbol", "UNKNOWN")

    # Get formatted amounts (with decimals), fall back to raw amounts
    amount_in_formatted = decoded_tx.get("amount_in_formatted")
    amount_out_formatted = decoded_tx.get("amount_out_formatted")

    if amount_in_formatted is not None and isinstance(
        amount_in_formatted, (int, float)
    ):
        amount_in = float(amount_in_formatted)
    else:
        raw_amount_in = decoded_tx.get("amount_in", 0)
        amount_in = (
            float(raw_amount_in) if isinstance(raw_amount_in, (int, float)) else 0.0
        )

    if amount_out_formatted is not None and isinstance(
        amount_out_formatted, (int, float)
    ):
        amount_out = float(amount_out_formatted)
    else:
        raw_amount_out = decoded_tx.get("amount_out", 0)
        amount_out = (
            float(raw_amount_out) if isinstance(raw_amount_out, (int, float)) else 0.0
        )

    return {
        "action": action,
        "assets": [token_in_symbol, token_out_symbol],
        "protocol": protocol,
        "outcome": outcome,
        "amounts": [amount_in, amount_out],
    }


def extract_intents_batch(decoded_txs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract intents from multiple decoded transactions in batch.

    Args:
        decoded_txs: List of decoded transaction dictionaries

    Returns:
        List of intent dictionaries

    Notes:
        - Skips transactions that fail to decode (logs warning)
        - Preserves order of input transactions
    """
    intents = []

    for i, decoded_tx in enumerate(decoded_txs):
        try:
            intent = extract_intent(decoded_tx)
            # Include transaction hash and block number for reference
            intent["tx_hash"] = decoded_tx.get("tx_hash", f"unknown_{i}")
            intent["block_number"] = decoded_tx.get("block_number", 0)
            intents.append(intent)
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Failed to extract intent from transaction "
                f"{decoded_tx.get('tx_hash', f'index_{i}')}: {e}"
            )
            continue

    logger.info(
        f"Extracted {len(intents)} intents from {len(decoded_txs)} transactions"
    )
    return intents
