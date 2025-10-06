#!/usr/bin/env python3
"""
CLI wrapper for decoding Ethereum transactions.

This script decodes raw transaction data into structured intents
using the decoders from eth_finetuning.extraction.decoders.

Usage:
    python scripts/decode_transactions.py \\
        --input data/raw/transactions.json \\
        --output data/processed/decoded.csv

RPC URL is read from configs/extraction_config.yaml by default.
"""

import json
import logging
import sys
from pathlib import Path

import click
import yaml

from eth_finetuning.extraction.core.utils import Web3ConnectionManager, setup_logging
from eth_finetuning.extraction.decoders.eth import decode_eth_transfer
from eth_finetuning.extraction.decoders.erc20 import decode_erc20_transfers
from eth_finetuning.extraction.decoders.uniswap import (
    decode_uniswap_v2_swaps,
    decode_uniswap_v3_swaps,
)
from eth_finetuning.extraction.export import export_to_csv

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


def load_transactions(input_path: Path) -> list[dict]:
    """Load transactions from JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)


def decode_transaction(
    tx: dict, receipt: dict, w3: Web3ConnectionManager
) -> list[dict]:
    """
    Decode a single transaction into structured intents.

    Args:
        tx: Transaction data
        receipt: Receipt data
        w3: Web3 connection manager

    Returns:
        List of decoded transaction dictionaries
    """
    decoded = []

    # Try ETH transfer decoder
    eth_result = decode_eth_transfer(tx, receipt)
    if eth_result:
        decoded.append(eth_result)

    # Try ERC-20 decoder
    erc20_results = decode_erc20_transfers(tx, receipt, w3.w3)
    if erc20_results:
        decoded.extend(erc20_results)

    # Try Uniswap V2 decoder
    uniswap_v2_results = decode_uniswap_v2_swaps(tx, receipt, w3.w3)
    if uniswap_v2_results:
        decoded.extend(uniswap_v2_results)

    # Try Uniswap V3 decoder
    uniswap_v3_results = decode_uniswap_v3_swaps(tx, receipt, w3.w3)
    if uniswap_v3_results:
        decoded.extend(uniswap_v3_results)

    return decoded


@click.command()
@click.option(
    "--input",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSON file with raw transaction data",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output CSV file path for decoded transactions",
)
@click.option(
    "--rpc-url",
    default=None,
    help="Ethereum RPC endpoint URL (overrides config file if provided)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to extraction config YAML file (default: configs/extraction_config.yaml)",
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
    input: Path, output: Path, rpc_url: str | None, config: Path | None, log_level: str
) -> None:
    """
    Decode Ethereum transactions into structured intents.

    Processes raw transaction data and applies decoders for:
    - ETH transfers
    - ERC-20 token transfers
    - Uniswap V2 swaps
    - Uniswap V3 swaps
    """
    # Load configuration
    cfg = load_config(config)

    setup_logging(level=log_level, log_format=cfg.get("logging", {}).get("format"))

    # Get RPC URL from CLI argument or config file
    if rpc_url is None:
        rpc_url = cfg.get("rpc", {}).get("endpoint")
        if not rpc_url or rpc_url == "PLACEHOLDER_RPC_URL":
            logger.error(
                "No RPC URL provided. Either:\n"
                "  1. Pass --rpc-url on command line, or\n"
                "  2. Set 'rpc.endpoint' in configs/extraction_config.yaml"
            )
            sys.exit(1)
        logger.info("Using RPC URL from config file")

    logger.info("Starting transaction decoding")
    logger.info(f"Input file: {input}")
    logger.info(f"Output file: {output}")
    logger.info(f"RPC URL: {rpc_url}")

    try:
        # Load raw transactions
        transactions = load_transactions(input)
        logger.info(f"Loaded {len(transactions)} transactions")

        # Initialize Web3 connection
        manager = Web3ConnectionManager(rpc_url=rpc_url)

        # Decode all transactions
        all_decoded = []
        for tx_data in transactions:
            # Extract transaction and receipt from combined data
            tx = {
                "hash": tx_data["tx_hash"],
                "from": tx_data["from"],
                "to": tx_data["to"],
                "value": tx_data["value"],
                "input": tx_data["input"],
                "blockNumber": tx_data["block_number"],
            }
            receipt = {
                "status": tx_data["status"],
                "logs": tx_data["logs"],
            }

            decoded = decode_transaction(tx, receipt, manager)
            all_decoded.extend(decoded)

        logger.info(f"Decoded {len(all_decoded)} transactions")

        # Export to CSV
        if all_decoded:
            export_to_csv(all_decoded, output)
            logger.info("Decoding completed successfully")
        else:
            logger.warning("No transactions were successfully decoded")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
