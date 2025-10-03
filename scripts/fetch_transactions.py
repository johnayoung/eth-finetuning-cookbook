#!/usr/bin/env python3
"""
Fetch Ethereum transactions from RPC endpoint.

CLI wrapper script for transaction fetching functionality.

Usage:
    python scripts/fetch_transactions.py \\
        --rpc-url https://mainnet.infura.io/v3/YOUR_KEY \\
        --tx-hashes transactions.txt \\
        --output data/raw/transactions.json

Transaction hashes file should contain one hash per line (with or without 0x prefix).
"""

import json
import logging
import sys
from pathlib import Path

import click
import yaml

from eth_finetuning.extraction.core.fetcher import (
    fetch_transactions_batch,
    load_transaction_hashes,
)
from eth_finetuning.extraction.core.utils import Web3ConnectionManager, setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path | None = None) -> dict:
    """Load extraction configuration from YAML file."""
    if config_path is None:
        # Default to project root configs/extraction_config.yaml
        project_root = Path(__file__).parent.parent
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


def save_transactions(
    transactions: list[dict], output_path: Path, pretty: bool = True
) -> None:
    """Save transaction data to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    help="Path to extraction config YAML file (optional)",
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
    and retries automatically.
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
        # Load transaction hashes
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

        # Fetch transactions
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
