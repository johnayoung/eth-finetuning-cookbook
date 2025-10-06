"""
Data normalization utilities for Ethereum transaction processing.

This module provides functions to normalize hex data fields when transitioning
between Web3.py native types (HexBytes), JSON storage, and decoder processing.

The normalization layer ensures:
- Consistent hex string format (with or without 0x prefix)
- Proper handling of Web3.py HexBytes objects
- Single source of truth for data format conversions
- Decoders don't need to handle format variations

Usage:
    from eth_finetuning.extraction.core.normalization import (
        normalize_hex_field,
        normalize_web3_transaction,
        web3_to_json,
    )

    # In decoders - parse hex fields consistently
    data_bytes = normalize_hex_field(log['data'])

    # In fetcher - serialize Web3 data for storage
    json_data = normalize_web3_transaction(web3_receipt)
"""

import json
import logging
from typing import Any

from hexbytes import HexBytes
from web3 import Web3

logger = logging.getLogger(__name__)


def normalize_hex_field(hex_string: str | HexBytes | bytes) -> bytes:
    """
    Normalize a hex field to bytes, handling various input formats.

    This function is the primary interface for decoders to parse hex data.
    It handles:
    - Strings with 0x prefix: "0x1234..."
    - Strings without 0x prefix: "1234..."
    - HexBytes objects (from Web3.py)
    - Raw bytes objects
    - Empty values: "0x", "", None

    Args:
        hex_string: Hex data in any supported format

    Returns:
        Raw bytes representation of the hex data

    Raises:
        ValueError: If the input cannot be parsed as hex data

    Examples:
        >>> normalize_hex_field("0x1234")
        b'\\x12\\x34'
        >>> normalize_hex_field("1234")
        b'\\x12\\x34'
        >>> normalize_hex_field(HexBytes("0x1234"))
        b'\\x12\\x34'
        >>> normalize_hex_field(b'\\x12\\x34')
        b'\\x12\\x34'
    """
    # Handle None or empty string
    if hex_string is None or hex_string == "" or hex_string == "0x":
        return b""

    # Already bytes
    if isinstance(hex_string, bytes):
        return hex_string

    # HexBytes object (from Web3.py)
    if isinstance(hex_string, HexBytes):
        return bytes(hex_string)

    # String - remove 0x prefix if present
    if isinstance(hex_string, str):
        hex_clean = hex_string[2:] if hex_string.startswith("0x") else hex_string

        # Validate it's valid hex
        if not hex_clean:
            return b""

        try:
            return bytes.fromhex(hex_clean)
        except ValueError as e:
            raise ValueError(f"Invalid hex string: {hex_string}") from e

    raise ValueError(f"Unsupported hex field type: {type(hex_string)}")


def normalize_hex_string(
    hex_data: str | HexBytes | bytes, with_prefix: bool = True
) -> str:
    """
    Normalize hex data to a consistent string format.

    Args:
        hex_data: Hex data in any supported format
        with_prefix: If True, include '0x' prefix in output

    Returns:
        Hex string in consistent format

    Examples:
        >>> normalize_hex_string("1234", with_prefix=True)
        '0x1234'
        >>> normalize_hex_string("0x1234", with_prefix=False)
        '1234'
        >>> normalize_hex_string(HexBytes("0x1234"))
        '0x1234'
    """
    # Convert to bytes first
    data_bytes = normalize_hex_field(hex_data)

    # Convert to hex string
    hex_str = data_bytes.hex()

    # Add prefix if requested
    return f"0x{hex_str}" if with_prefix else hex_str


def web3_json_encoder(obj: Any) -> Any:
    """
    JSON encoder for Web3.py objects.

    Handles serialization of:
    - HexBytes → hex string with 0x prefix
    - bytes → hex string with 0x prefix
    - Other Web3.py AttributeDict types

    This ensures consistent JSON output that matches Web3.py conventions.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation

    Raises:
        TypeError: If object type is not supported
    """
    if isinstance(obj, HexBytes):
        # HexBytes.hex() returns without 0x prefix, so add it
        return "0x" + obj.hex()

    if isinstance(obj, bytes):
        return "0x" + obj.hex()

    # Let Web3's native encoder handle other types
    try:
        return Web3.to_json(obj)
    except TypeError:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def normalize_web3_transaction(tx_data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize Web3 transaction data for consistent JSON storage.

    Converts all HexBytes fields to hex strings with 0x prefix.
    This ensures that when the data is loaded from JSON, decoders
    can rely on consistent formatting.

    Args:
        tx_data: Transaction or receipt data from Web3.py

    Returns:
        Normalized dictionary with consistent hex string formatting

    Note:
        This function recursively processes nested structures (logs, etc.)
    """
    normalized = {}

    for key, value in tx_data.items():
        if isinstance(value, HexBytes):
            # Convert to hex string with 0x prefix (HexBytes.hex() doesn't include 0x)
            normalized[key] = "0x" + value.hex()
        elif isinstance(value, bytes):
            # Convert raw bytes to hex string with 0x prefix
            normalized[key] = "0x" + value.hex()
        elif isinstance(value, list):
            # Recursively normalize lists (e.g., logs, topics)
            normalized[key] = [
                (
                    normalize_web3_transaction(item)
                    if isinstance(item, dict)
                    else (
                        "0x" + item.hex()
                        if isinstance(item, (HexBytes, bytes))
                        else item
                    )
                )
                for item in value
            ]
        elif isinstance(value, dict):
            # Recursively normalize nested dicts
            normalized[key] = normalize_web3_transaction(value)
        else:
            # Keep other types as-is
            normalized[key] = value

    return normalized


def normalize_log(log: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a single log entry for consistent processing.

    Ensures topics and data fields have consistent hex format.
    This is a convenience function for decoder implementations.

    Args:
        log: Log entry from transaction receipt

    Returns:
        Normalized log with consistent hex formatting
    """
    normalized = log.copy()

    # Normalize topics list
    if "topics" in normalized:
        normalized["topics"] = [
            normalize_hex_string(topic, with_prefix=True)
            for topic in normalized["topics"]
        ]

    # Normalize data field
    if "data" in normalized:
        normalized["data"] = normalize_hex_string(normalized["data"], with_prefix=True)

    return normalized


def parse_log_data(log: dict[str, Any]) -> bytes:
    """
    Extract and parse log data field as bytes.

    Convenience function for decoders to get log data in bytes format.

    Args:
        log: Log entry with 'data' field

    Returns:
        Raw bytes from log data field

    Raises:
        ValueError: If log has no data field or data is invalid
    """
    data = log.get("data")
    if data is None:
        raise ValueError("Log has no 'data' field")

    return normalize_hex_field(data)


def parse_log_topics(log: dict[str, Any]) -> list[bytes]:
    """
    Extract and parse log topics as list of bytes.

    Convenience function for decoders to get log topics in bytes format.

    Args:
        log: Log entry with 'topics' field

    Returns:
        List of raw bytes from log topics

    Raises:
        ValueError: If log has no topics field
    """
    topics = log.get("topics")
    if topics is None:
        raise ValueError("Log has no 'topics' field")

    return [normalize_hex_field(topic) for topic in topics]
