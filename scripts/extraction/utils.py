"""
Utility functions for Ethereum data extraction.

This module provides:
- Web3 connection management with retry logic
- ABI loading from JSON files
- Error handling patterns for RPC interactions
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from web3 import Web3
from web3.exceptions import Web3Exception
import requests.exceptions

# Configure module logger
logger = logging.getLogger(__name__)

# Type variable for generic retry decorator
T = TypeVar("T")


class Web3ConnectionManager:
    """
    Manages Web3 connection with automatic retry logic and error handling.

    Implements exponential backoff for transient RPC failures and provides
    a consistent interface for blockchain interactions.
    """

    def __init__(
        self,
        rpc_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        """
        Initialize Web3 connection manager.

        Args:
            rpc_url: Ethereum RPC endpoint URL (e.g., Infura, Alchemy)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            backoff_factor: Exponential backoff multiplier (delay = backoff_factor^attempt)

        Raises:
            ValueError: If RPC URL is invalid or connection cannot be established
        """
        if not rpc_url or rpc_url == "PLACEHOLDER_RPC_URL":
            raise ValueError(
                "Invalid RPC URL. Please configure a valid Ethereum RPC endpoint "
                "in configs/extraction_config.yaml or pass via --rpc-url"
            )

        self.rpc_url = rpc_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Initialize Web3 provider with timeout
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))

        # Verify connection
        if not self.w3.is_connected():
            raise ValueError(f"Failed to connect to Ethereum node at {rpc_url}")

        logger.info(f"Connected to Ethereum node at {rpc_url}")

    def retry_with_backoff(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute function with exponential backoff retry logic.

        Retries the function on transient errors (network issues, rate limits)
        with exponentially increasing delays between attempts.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from successful function execution

        Raises:
            Exception: Re-raises the last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)

            except (
                requests.exceptions.RequestException,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                Web3Exception,
            ) as e:
                last_exception = e

                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = self.backoff_factor**attempt
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Request failed after {self.max_retries} attempts: {e}"
                    )

        # All retries exhausted - raise the last exception encountered
        if last_exception:
            raise last_exception
        raise RuntimeError("Function failed but no exception was captured")

    def get_transaction(self, tx_hash: str) -> dict[str, Any]:
        """
        Fetch transaction data with automatic retry.

        Args:
            tx_hash: Transaction hash (with or without 0x prefix)

        Returns:
            Transaction data dictionary

        Raises:
            ValueError: If transaction hash is invalid
            Exception: If transaction cannot be fetched after all retries
        """
        # Ensure transaction hash has 0x prefix
        if not tx_hash.startswith("0x"):
            tx_hash = "0x" + tx_hash

        logger.debug(f"Fetching transaction: {tx_hash}")

        tx = self.retry_with_backoff(self.w3.eth.get_transaction, tx_hash)

        # Convert HexBytes to hex strings for JSON serialization
        return self._serialize_transaction(tx)

    def get_transaction_receipt(self, tx_hash: str) -> dict[str, Any]:
        """
        Fetch transaction receipt with automatic retry.

        Args:
            tx_hash: Transaction hash (with or without 0x prefix)

        Returns:
            Transaction receipt dictionary including logs

        Raises:
            ValueError: If transaction hash is invalid
            Exception: If receipt cannot be fetched after all retries
        """
        # Ensure transaction hash has 0x prefix
        if not tx_hash.startswith("0x"):
            tx_hash = "0x" + tx_hash

        logger.debug(f"Fetching transaction receipt: {tx_hash}")

        receipt = self.retry_with_backoff(self.w3.eth.get_transaction_receipt, tx_hash)

        # Convert HexBytes to hex strings for JSON serialization
        return self._serialize_receipt(receipt)

    def get_block(self, block_identifier: int | str) -> dict[str, Any]:
        """
        Fetch block data with automatic retry.

        Args:
            block_identifier: Block number (int) or hash (str)

        Returns:
            Block data dictionary

        Raises:
            Exception: If block cannot be fetched after all retries
        """
        logger.debug(f"Fetching block: {block_identifier}")

        block = self.retry_with_backoff(self.w3.eth.get_block, block_identifier)

        return self._serialize_block(block)

    @staticmethod
    def _serialize_transaction(tx: Any) -> dict[str, Any]:
        """
        Convert Web3 transaction object to JSON-serializable dict.

        Handles HexBytes conversion and ensures all addresses are checksummed.
        """
        return {
            "hash": tx["hash"].hex() if hasattr(tx["hash"], "hex") else tx["hash"],
            "nonce": tx["nonce"],
            "blockHash": (
                tx["blockHash"].hex()
                if tx["blockHash"] and hasattr(tx["blockHash"], "hex")
                else None
            ),
            "blockNumber": tx["blockNumber"],
            "transactionIndex": tx["transactionIndex"],
            "from": Web3.to_checksum_address(tx["from"]),
            "to": Web3.to_checksum_address(tx["to"]) if tx["to"] else None,
            "value": tx["value"],
            "gas": tx["gas"],
            "gasPrice": tx["gasPrice"],
            "input": tx["input"].hex() if hasattr(tx["input"], "hex") else tx["input"],
            "type": tx.get("type"),
            "chainId": tx.get("chainId"),
        }

    @staticmethod
    def _serialize_receipt(receipt: Any) -> dict[str, Any]:
        """
        Convert Web3 receipt object to JSON-serializable dict.

        Includes log decoding and status information.
        """
        return {
            "transactionHash": (
                receipt["transactionHash"].hex()
                if hasattr(receipt["transactionHash"], "hex")
                else receipt["transactionHash"]
            ),
            "transactionIndex": receipt["transactionIndex"],
            "blockHash": (
                receipt["blockHash"].hex()
                if hasattr(receipt["blockHash"], "hex")
                else receipt["blockHash"]
            ),
            "blockNumber": receipt["blockNumber"],
            "from": Web3.to_checksum_address(receipt["from"]),
            "to": Web3.to_checksum_address(receipt["to"]) if receipt["to"] else None,
            "contractAddress": (
                Web3.to_checksum_address(receipt["contractAddress"])
                if receipt.get("contractAddress")
                else None
            ),
            "cumulativeGasUsed": receipt["cumulativeGasUsed"],
            "gasUsed": receipt["gasUsed"],
            "effectiveGasPrice": receipt.get("effectiveGasPrice"),
            "status": receipt["status"],
            "logs": [
                {
                    "address": Web3.to_checksum_address(log["address"]),
                    "topics": [
                        topic.hex() if hasattr(topic, "hex") else topic
                        for topic in log["topics"]
                    ],
                    "data": (
                        log["data"].hex()
                        if hasattr(log["data"], "hex")
                        else log["data"]
                    ),
                    "blockNumber": log["blockNumber"],
                    "transactionHash": (
                        log["transactionHash"].hex()
                        if hasattr(log["transactionHash"], "hex")
                        else log["transactionHash"]
                    ),
                    "transactionIndex": log["transactionIndex"],
                    "logIndex": log["logIndex"],
                }
                for log in receipt["logs"]
            ],
        }

    @staticmethod
    def _serialize_block(block: Any) -> dict[str, Any]:
        """Convert Web3 block object to JSON-serializable dict."""
        return {
            "number": block["number"],
            "hash": (
                block["hash"].hex() if hasattr(block["hash"], "hex") else block["hash"]
            ),
            "timestamp": block["timestamp"],
            "parentHash": (
                block["parentHash"].hex()
                if hasattr(block["parentHash"], "hex")
                else block["parentHash"]
            ),
            "miner": Web3.to_checksum_address(block["miner"]),
            "gasLimit": block["gasLimit"],
            "gasUsed": block["gasUsed"],
        }


def load_abi(abi_name: str, abis_dir: Path | None = None) -> list[dict[str, Any]]:
    """
    Load ABI from JSON file.

    ABIs are stored in scripts/extraction/abis/ directory by default.
    Standard ABIs included: erc20.json, uniswap_v2.json, uniswap_v3.json

    Args:
        abi_name: Name of ABI file (with or without .json extension)
        abis_dir: Optional custom directory path. Defaults to scripts/extraction/abis/

    Returns:
        ABI as list of dictionaries

    Raises:
        FileNotFoundError: If ABI file does not exist
        json.JSONDecodeError: If ABI file is not valid JSON

    Example:
        >>> erc20_abi = load_abi('erc20')
        >>> contract = w3.eth.contract(address=token_address, abi=erc20_abi)
    """
    # Default to scripts/extraction/abis/ directory
    if abis_dir is None:
        abis_dir = Path(__file__).parent / "abis"

    # Add .json extension if not present
    if not abi_name.endswith(".json"):
        abi_name += ".json"

    abi_path = abis_dir / abi_name

    if not abi_path.exists():
        raise FileNotFoundError(
            f"ABI file not found: {abi_path}. "
            f"Please ensure the ABI is saved in {abis_dir}"
        )

    try:
        with open(abi_path, "r") as f:
            abi = json.load(f)

        logger.debug(f"Loaded ABI from {abi_path}")
        return abi

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in ABI file {abi_path}: {e}")
        raise


def setup_logging(level: str = "INFO", log_format: str | None = None) -> None:
    """
    Configure logging for extraction scripts.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string. Uses default if None.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from web3 and urllib3 loggers
    logging.getLogger("web3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
