"""
Prompt templates for instruction-tuning format.

This module provides Alpaca-style templates for converting transaction data
and intents into instruction-tuning format for HuggingFace Trainer.

Template Format (Alpaca):
    {
        "instruction": "Task description for the model",
        "input": "Input data (transaction details)",
        "output": "Expected output (structured intent JSON)"
    }

Usage:
    from eth_finetuning.dataset.templates import format_training_example

    example = format_training_example(decoded_tx, intent)
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default instruction template
DEFAULT_INSTRUCTION = (
    "Extract the structured intent from this Ethereum transaction. "
    "Identify the action, involved assets, protocol, outcome, and amounts."
)


def format_training_example(
    decoded_tx: dict[str, Any],
    intent: dict[str, Any],
    instruction: str | None = None,
) -> dict[str, Any]:
    """
    Format a training example in Alpaca instruction-tuning format.

    Converts decoded transaction data and extracted intent into a structured
    prompt suitable for fine-tuning language models.

    Args:
        decoded_tx: Decoded transaction dictionary from decoders
        intent: Extracted intent dictionary from intent_extraction.py
        instruction: Optional custom instruction (uses default if None)

    Returns:
        Training example dictionary with structure:
        {
            "instruction": str - Task description
            "input": str - JSON string of transaction data
            "output": str - JSON string of intent data
        }

    Raises:
        ValueError: If decoded_tx or intent is empty or missing required fields

    Notes:
        - Input is formatted as clean JSON (minified, no extra whitespace)
        - Output is formatted as clean JSON (minified, no extra whitespace)
        - Only includes essential fields in input to reduce token count
        - Instruction is consistent across all examples for better learning
    """
    if not decoded_tx:
        raise ValueError("decoded_tx cannot be empty")
    if not intent:
        raise ValueError("intent cannot be empty")

    # Use default instruction if not provided
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION

    # Format input: select essential fields from decoded transaction
    input_data = _format_transaction_input(decoded_tx)

    # Format output: clean intent JSON
    output_data = _format_intent_output(intent)

    return {
        "instruction": instruction,
        "input": input_data,
        "output": output_data,
    }


def _format_transaction_input(decoded_tx: dict[str, Any]) -> str:
    """
    Format decoded transaction as JSON string for model input.

    Extracts essential fields to minimize token count while preserving
    all information needed to extract intent.

    Args:
        decoded_tx: Decoded transaction dictionary

    Returns:
        JSON string with essential transaction fields
    """
    protocol = decoded_tx.get("protocol", "unknown")

    # Base fields common to all protocols
    input_dict = {
        "tx_hash": decoded_tx.get("tx_hash", "unknown"),
        "protocol": protocol,
        "action": decoded_tx.get("action", "unknown"),
        "status": decoded_tx.get("status", "success"),
    }

    # Protocol-specific fields
    if protocol == "ethereum":
        input_dict.update(
            {
                "from": decoded_tx.get("from", ""),
                "to": decoded_tx.get("to", ""),
                "amount_wei": decoded_tx.get("amount_wei", 0),
                "amount_eth": decoded_tx.get("amount_eth", 0.0),
            }
        )
    elif protocol == "erc20":
        input_dict.update(
            {
                "from": decoded_tx.get("from", ""),
                "to": decoded_tx.get("to", ""),
                "token_address": decoded_tx.get("token_address", ""),
                "token_symbol": decoded_tx.get("token_symbol", "UNKNOWN"),
                "amount": decoded_tx.get("amount", 0),
                "amount_formatted": decoded_tx.get("amount_formatted"),
                "token_decimals": decoded_tx.get("token_decimals"),
            }
        )
    elif protocol in ["uniswap_v2", "uniswap_v3"]:
        input_dict.update(
            {
                "pool_address": decoded_tx.get("pool_address", ""),
                "token_in": decoded_tx.get("token_in", ""),
                "token_out": decoded_tx.get("token_out", ""),
                "token_in_symbol": decoded_tx.get("token_in_symbol", "UNKNOWN"),
                "token_out_symbol": decoded_tx.get("token_out_symbol", "UNKNOWN"),
                "amount_in": decoded_tx.get("amount_in", 0),
                "amount_out": decoded_tx.get("amount_out", 0),
                "amount_in_formatted": decoded_tx.get("amount_in_formatted"),
                "amount_out_formatted": decoded_tx.get("amount_out_formatted"),
            }
        )

        # V3-specific fields
        if protocol == "uniswap_v3":
            input_dict.update(
                {
                    "tick": decoded_tx.get("tick"),
                    "sqrt_price_x96": decoded_tx.get("sqrt_price_x96"),
                    "liquidity": decoded_tx.get("liquidity"),
                }
            )

    # Convert to JSON string (minified for token efficiency)
    return json.dumps(input_dict, separators=(",", ":"))


def _format_intent_output(intent: dict[str, Any]) -> str:
    """
    Format intent as JSON string for model output.

    Creates clean intent JSON without tx_hash and block_number
    (reference fields, not part of intent structure).

    Args:
        intent: Intent dictionary from intent_extraction.py

    Returns:
        JSON string with intent fields
    """
    # Extract intent fields (exclude reference fields)
    output_dict = {
        "action": intent.get("action", "unknown"),
        "assets": intent.get("assets", []),
        "protocol": intent.get("protocol", "unknown"),
        "outcome": intent.get("outcome", "success"),
        "amounts": intent.get("amounts", []),
    }

    # Convert to JSON string (minified for token efficiency)
    return json.dumps(output_dict, separators=(",", ":"))


def format_training_examples_batch(
    decoded_txs: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    instruction: str | None = None,
) -> list[dict[str, Any]]:
    """
    Format multiple training examples in batch.

    Args:
        decoded_txs: List of decoded transaction dictionaries
        intents: List of intent dictionaries (must match decoded_txs length)
        instruction: Optional custom instruction (uses default if None)

    Returns:
        List of training example dictionaries

    Raises:
        ValueError: If decoded_txs and intents have different lengths

    Notes:
        - Skips examples that fail to format (logs warning)
        - Preserves order of input transactions
    """
    if len(decoded_txs) != len(intents):
        raise ValueError(
            f"decoded_txs and intents must have same length "
            f"(got {len(decoded_txs)} and {len(intents)})"
        )

    examples = []

    for i, (decoded_tx, intent) in enumerate(zip(decoded_txs, intents)):
        try:
            example = format_training_example(decoded_tx, intent, instruction)
            examples.append(example)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to format training example at index {i}: {e}")
            continue

    logger.info(
        f"Formatted {len(examples)} training examples from {len(decoded_txs)} transactions"
    )
    return examples


def format_inference_prompt(decoded_tx: dict[str, Any]) -> str:
    """
    Format a transaction for inference (without ground truth intent).

    Useful for generating prompts to test the fine-tuned model on new transactions.

    Args:
        decoded_tx: Decoded transaction dictionary

    Returns:
        Formatted prompt string combining instruction and input

    Notes:
        - Follows same format as training examples
        - Can be used directly with model.generate()
    """
    instruction = DEFAULT_INSTRUCTION
    input_data = _format_transaction_input(decoded_tx)

    # Format as instruction-input prompt (common format for Alpaca-style models)
    prompt = f"{instruction}\n\nInput:\n{input_data}\n\nOutput:\n"

    return prompt
