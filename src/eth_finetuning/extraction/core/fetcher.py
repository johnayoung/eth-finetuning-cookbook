"""
Transaction fetching utilities.

This module provides functions for fetching Ethereum transactions from RPC endpoints
with automatic rate limiting, retry logic, and parallel processing.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .utils import Web3ConnectionManager

logger = logging.getLogger(__name__)


def load_transaction_hashes(file_path: Path) -> list[str]:
    """
    Load transaction hashes from text file.

    Args:
        file_path: Path to file containing transaction hashes (one per line)

    Returns:
        List of transaction hashes (normalized with 0x prefix)

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Transaction hashes file not found: {file_path}")

    hashes = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Validate hex format (with or without 0x prefix)
            tx_hash = line if line.startswith("0x") else "0x" + line

            # Basic validation: should be 66 characters (0x + 64 hex chars)
            if len(tx_hash) != 66:
                logger.warning(
                    f"Line {line_num}: Invalid transaction hash length: {line} "
                    f"(expected 66 chars, got {len(tx_hash)})"
                )
                continue

            try:
                # Validate hex format
                int(tx_hash, 16)
                hashes.append(tx_hash)
            except ValueError:
                logger.warning(f"Line {line_num}: Invalid hex format: {line}")
                continue

    logger.info(f"Loaded {len(hashes)} transaction hashes from {file_path}")
    return hashes


def fetch_transaction_data(
    tx_hash: str, manager: Web3ConnectionManager
) -> dict[str, Any] | None:
    """
    Fetch complete transaction data including receipt and block info.

    Args:
        tx_hash: Transaction hash
        manager: Web3 connection manager

    Returns:
        Dictionary with transaction, receipt, and block data, or None if failed
    """
    try:
        # Fetch transaction and receipt
        tx = manager.get_transaction(tx_hash)
        receipt = manager.get_transaction_receipt(tx_hash)

        # Combine transaction and receipt data
        # Note: tx and receipt are already serialized by Web3ConnectionManager
        # with proper 0x prefixes using Web3.to_json()
        result = {
            "tx_hash": tx_hash,
            "block_number": tx["blockNumber"],
            "from": tx["from"],
            "to": tx["to"],
            "value": tx["value"],
            "input": tx["input"],
            "gas": tx["gas"],
            "gas_price": tx["gasPrice"],
            "nonce": tx["nonce"],
            "transaction_index": tx["transactionIndex"],
            "status": receipt["status"],
            "gas_used": receipt["gasUsed"],
            "logs": receipt["logs"],
            "contract_address": receipt.get("contractAddress"),
        }

        logger.info(f"Successfully fetched transaction {tx_hash[:10]}...")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch transaction {tx_hash}: {e}")
        return None


def fetch_transactions_batch(
    tx_hashes: list[str],
    manager: Web3ConnectionManager,
    batch_size: int = 10,
    max_workers: int = 5,
    rate_limit_delay: float = 0.2,
) -> list[dict[str, Any]]:
    """
    Fetch multiple transactions with rate limiting and parallel processing.

    Args:
        tx_hashes: List of transaction hashes to fetch
        manager: Web3 connection manager
        batch_size: Number of transactions to process in parallel
        max_workers: Maximum number of concurrent workers
        rate_limit_delay: Delay between batches in seconds

    Returns:
        List of successfully fetched transaction data dictionaries
    """
    results = []
    failed_count = 0

    # Process in batches to respect rate limits
    for i in range(0, len(tx_hashes), batch_size):
        batch = tx_hashes[i : i + batch_size]
        logger.info(
            f"Processing batch {i // batch_size + 1}/{(len(tx_hashes) + batch_size - 1) // batch_size} "
            f"({len(batch)} transactions)"
        )

        # Fetch transactions in parallel within batch
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_transaction_data, tx_hash, manager): tx_hash
                for tx_hash in batch
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_count += 1

        # Rate limiting delay between batches
        if i + batch_size < len(tx_hashes):
            time.sleep(rate_limit_delay)

    logger.info(
        f"Completed fetching {len(results)} transactions " f"({failed_count} failed)"
    )

    return results
