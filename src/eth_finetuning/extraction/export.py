"""
Export decoded transactions to CSV format.

This module provides utilities for exporting decoded transaction data
(ETH transfers, ERC-20 transfers) to CSV files for further analysis.

Usage:
    from eth_finetuning.extraction.export import export_to_csv

    export_to_csv(decoded_transactions, "data/processed/transfers.csv")
"""

import csv
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


def export_to_csv(
    decoded_transactions: list[dict[str, Any]],
    output_path: str | Path,
    timestamp_getter: Callable[[int], int] | None = None,
) -> None:
    """
    Export decoded transactions to CSV file.

    Creates a CSV with columns: tx_hash, block, timestamp, from, to, value,
    decoded_action, protocol, assets, amounts, pool_address, token_in, token_out,
    amount_in, amount_out.

    Args:
        decoded_transactions: List of decoded transaction dictionaries
            Can contain mix of ETH transfers and ERC-20 transfers
        output_path: Path to output CSV file (will create parent directories)
        timestamp_getter: Optional function to fetch block timestamps
            Signature: timestamp_getter(block_number: int) -> int
            If None, timestamp column will be empty

    Raises:
        ValueError: If decoded_transactions is empty
        IOError: If file cannot be written

    Notes:
        - Handles mixed transaction types (ETH and ERC-20)
        - Creates parent directories if they don't exist
        - Overwrites existing file at output_path
    """
    if not decoded_transactions:
        raise ValueError("Cannot export empty transaction list")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert decoded transactions to CSV rows
    rows = []
    for tx in decoded_transactions:
        row = _transaction_to_csv_row(tx, timestamp_getter)
        rows.append(row)

    # Create DataFrame for better CSV handling
    df = pd.DataFrame(rows)

    # Write to CSV
    df.to_csv(output_path, index=False)

    logger.info(f"Exported {len(rows)} transactions to {output_path}")


def _transaction_to_csv_row(
    tx: dict[str, Any], timestamp_getter: Callable[[int], int] | None = None
) -> dict[str, Any]:
    """
    Convert decoded transaction to CSV row dictionary.

    Handles both ETH transfers and ERC-20 transfers with unified schema.

    Args:
        tx: Decoded transaction dictionary
        timestamp_getter: Optional function to fetch block timestamp

    Returns:
        Dictionary with columns: tx_hash, block, timestamp, from, to, value,
        decoded_action, protocol, assets, amounts
    """
    # Extract common fields
    tx_hash = tx.get("tx_hash", "")
    block = tx.get("block_number", 0)
    from_address = tx.get("from", "")
    to_address = tx.get("to", "")
    action = tx.get("action", "")
    protocol = tx.get("protocol", "")

    # Fetch timestamp if getter provided
    timestamp = ""
    if timestamp_getter and block:
        try:
            timestamp = timestamp_getter(block)
        except Exception as e:
            logger.warning(f"Failed to fetch timestamp for block {block}: {e}")

    # Protocol-specific field extraction
    if protocol == "ethereum":
        # ETH transfer
        value = tx.get("amount_eth", 0)
        assets = "ETH"
        amounts = str(value)
        pool_address = ""
        token_in = ""
        token_out = ""
        amount_in = ""
        amount_out = ""

    elif protocol == "erc20":
        # ERC-20 transfer
        token_symbol = tx.get("token_symbol", "UNKNOWN")
        amount_formatted = tx.get("amount_formatted")
        amount_raw = tx.get("amount", 0)

        # Use formatted amount if available, otherwise raw amount
        if amount_formatted is not None:
            value = amount_formatted
            amounts = str(amount_formatted)
        else:
            value = amount_raw
            amounts = str(amount_raw)

        assets = f"{token_symbol} ({tx.get('token_address', 'unknown')})"
        pool_address = ""
        token_in = ""
        token_out = ""
        amount_in = ""
        amount_out = ""

    elif protocol in ["uniswap_v2", "uniswap_v3"]:
        # Uniswap swap
        token_in_symbol = tx.get("token_in_symbol", "UNKNOWN")
        token_out_symbol = tx.get("token_out_symbol", "UNKNOWN")
        amount_in_formatted = tx.get("amount_in_formatted")
        amount_out_formatted = tx.get("amount_out_formatted")
        amount_in_raw = tx.get("amount_in", 0)
        amount_out_raw = tx.get("amount_out", 0)

        # Use formatted amounts if available
        if amount_in_formatted is not None:
            amount_in_display = amount_in_formatted
        else:
            amount_in_display = amount_in_raw

        if amount_out_formatted is not None:
            amount_out_display = amount_out_formatted
        else:
            amount_out_display = amount_out_raw

        value = f"{amount_in_display} {token_in_symbol} → {amount_out_display} {token_out_symbol}"
        assets = f"{token_in_symbol} → {token_out_symbol}"
        amounts = f"{amount_in_display} → {amount_out_display}"
        pool_address = tx.get("pool_address", "")
        token_in = tx.get("token_in", "")
        token_out = tx.get("token_out", "")
        amount_in = str(amount_in_display)
        amount_out = str(amount_out_display)

        # Override from/to for swaps (use sender/recipient)
        from_address = tx.get("sender", from_address)
        to_address = tx.get("recipient", to_address)

    else:
        # Unknown protocol
        value = tx.get("value", 0)
        assets = ""
        amounts = ""
        pool_address = ""
        token_in = ""
        token_out = ""
        amount_in = ""
        amount_out = ""

    return {
        "tx_hash": tx_hash,
        "block": block,
        "timestamp": timestamp,
        "from": from_address,
        "to": to_address,
        "value": value,
        "decoded_action": action,
        "protocol": protocol,
        "assets": assets,
        "amounts": amounts,
        "status": tx.get("status", "unknown"),
        "pool_address": pool_address,
        "token_in": token_in,
        "token_out": token_out,
        "amount_in": amount_in,
        "amount_out": amount_out,
    }


def merge_and_export(
    eth_transfers: list[dict[str, Any]],
    erc20_transfers: list[dict[str, Any]],
    output_path: str | Path,
    timestamp_getter: Callable[[int], int] | None = None,
) -> None:
    """
    Merge ETH and ERC-20 transfers and export to single CSV.

    Convenience function for exporting mixed transaction types.

    Args:
        eth_transfers: List of decoded ETH transfers
        erc20_transfers: List of decoded ERC-20 transfers
        output_path: Path to output CSV file
        timestamp_getter: Optional function to fetch block timestamps

    Raises:
        ValueError: If both lists are empty
        IOError: If file cannot be written
    """
    # Merge lists
    all_transactions = eth_transfers + erc20_transfers

    if not all_transactions:
        raise ValueError("No transactions to export")

    # Sort by block number for chronological order
    all_transactions.sort(key=lambda tx: tx.get("block_number", 0))

    # Export to CSV
    export_to_csv(all_transactions, output_path, timestamp_getter)

    logger.info(
        f"Merged and exported {len(eth_transfers)} ETH transfers and "
        f"{len(erc20_transfers)} ERC-20 transfers to {output_path}"
    )


def export_with_web3(
    decoded_transactions: list[dict[str, Any]],
    output_path: str | Path,
    w3: Any,  # Web3 instance
) -> None:
    """
    Export decoded transactions to CSV with block timestamps from Web3.

    Convenience wrapper that fetches block timestamps using Web3 connection.

    Args:
        decoded_transactions: List of decoded transaction dictionaries
        output_path: Path to output CSV file
        w3: Web3 instance (or Web3ConnectionManager) for fetching block data

    Raises:
        ValueError: If decoded_transactions is empty
        IOError: If file cannot be written

    Notes:
        - Makes RPC calls to fetch block timestamps (can be slow for many blocks)
        - Caches block timestamps to avoid duplicate RPC calls
    """
    # Cache block timestamps to avoid duplicate RPC calls
    block_cache: dict[int, int] = {}

    def get_timestamp(block_number: int) -> int:
        """Fetch block timestamp with caching."""
        if block_number not in block_cache:
            # Fetch block data
            if hasattr(w3, "get_block"):
                # Web3ConnectionManager instance
                block = w3.get_block(block_number)
            else:
                # Raw Web3 instance
                block = w3.eth.get_block(block_number)

            block_cache[block_number] = block.get("timestamp", 0)

        return block_cache[block_number]

    export_to_csv(decoded_transactions, output_path, timestamp_getter=get_timestamp)
