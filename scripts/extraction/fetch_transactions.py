#!/usr/bin/env python3
"""
Fetch Ethereum transactions from RPC endpoint.

This script fetches transaction data and receipts from an Ethereum node,
handling rate limits and retries automatically. Outputs raw transaction
data in JSON format for further processing by decoders.

Usage:
    python fetch_transactions.py \\
        --rpc-url https://mainnet.infura.io/v3/YOUR_KEY \\
        --tx-hashes transactions.txt \\
        --output data/raw/transactions.json

Transaction hashes file should contain one hash per line (with or without 0x prefix).
"""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click
import yaml

# Relative import from same package
from .utils import Web3ConnectionManager, setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load extraction configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to configs/extraction_config.yaml

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to project root configs/extraction_config.yaml
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "extraction_config.yaml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        sys.exit(1)


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


def save_transactions(
    transactions: list[dict[str, Any]], output_path: Path, pretty: bool = True
) -> None:
    """
    Save transaction data to JSON file.

    Args:
        transactions: List of transaction data dictionaries
        output_path: Output file path
        pretty: Whether to pretty-print JSON (default True)
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_path, "w") as f:
        if pretty:
            json.dump(transactions, f, indent=2)
        else:
            json.dump(transactions, f)

    logger.info(f"Saved {len(transactions)} transactions to {output_path}")


@click.command()
@click.option(
    "--rpc-url",
    required=True,
    help="Ethereum RPC endpoint URL (e.g., https://mainnet.infura.io/v3/YOUR_KEY)",
)
@click.option(
    "--tx-hashes",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to file containing transaction hashes (one per line)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSON file path for transaction data",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to extraction config YAML file (optional, uses default if not provided)",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Logging level",
)
def main(
    rpc_url: str, tx_hashes: Path, output: Path, config: Path | None, log_level: str
) -> None:
    """
    Fetch Ethereum transactions from RPC endpoint.
    
    This script fetches transaction data and receipts, handling rate limits
    and retries automatically. Output includes transaction details, receipts,
    and logs in JSON format.
    
    Example:
        python fetch_transactions.py \\
            --rpc-url https://mainnet.infura.io/v3/YOUR_KEY \\
            --tx-hashes transactions.txt \\
            --output data/raw/transactions.json
    """
    # Load configuration
    cfg = load_config(config)

    # Setup logging
    setup_logging(level=log_level, log_format=cfg.get("logging", {}).get("format"))

    logger.info("Starting transaction fetch process")
    logger.info(f"RPC URL: {rpc_url}")
    logger.info(f"Transaction hashes file: {tx_hashes}")
    logger.info(f"Output file: {output}")

    try:
        # Load transaction hashes from file
        hashes = load_transaction_hashes(tx_hashes)

        if not hashes:
            logger.error("No valid transaction hashes found in input file")
            sys.exit(1)

        # Initialize Web3 connection manager
        rpc_config = cfg.get("rpc", {})
        manager = Web3ConnectionManager(
            rpc_url=rpc_url,
            timeout=rpc_config.get("timeout", 30),
            max_retries=rpc_config.get("rate_limit", {}).get("max_retries", 3),
            backoff_factor=rpc_config.get("rate_limit", {}).get("backoff_factor", 2.0),
        )

        # Fetch transactions with rate limiting
        batch_config = cfg.get("batch", {})
        transactions = fetch_transactions_batch(
            tx_hashes=hashes,
            manager=manager,
            batch_size=batch_config.get("batch_size", 10),
            max_workers=batch_config.get("max_workers", 5),
            rate_limit_delay=rpc_config.get("rate_limit", {}).get("delay", 0.2),
        )

        if not transactions:
            logger.error("Failed to fetch any transactions")
            sys.exit(1)

        # Save results
        output_config = cfg.get("output", {})
        save_transactions(
            transactions=transactions,
            output_path=output,
            pretty=output_config.get("pretty", True),
        )

        logger.info("Transaction fetch completed successfully")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
