#!/usr/bin/env python3
"""
Decode ETH transfer transactions.

This module extracts structured intent data from simple Ethereum transfers
(native ETH transfers without contract interactions).

Usage:
    from scripts.extraction.decode_eth_transfers import decode_eth_transfer

    decoded = decode_eth_transfer(transaction, receipt)
"""

import logging
from typing import Any

from web3 import Web3

logger = logging.getLogger(__name__)


def decode_eth_transfer(
    transaction: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Decode a simple ETH transfer transaction.

    Extracts structured intent data from native ETH transfers. Only processes
    transactions with no input data (simple transfers, not contract calls).

    Args:
        transaction: Transaction data containing from, to, value, input fields
        receipt: Transaction receipt containing status, gasUsed fields

    Returns:
        Decoded intent dictionary with structure:
        {
            "action": "transfer",
            "protocol": "ethereum",
            "from": checksummed address,
            "to": checksummed address,
            "amount_wei": int (value in Wei),
            "amount_eth": float (value in Ether),
            "status": "success" | "failed",
            "gas_used": int
        }
        Returns None if transaction is not a simple ETH transfer

    Notes:
        - Handles zero-value transfers (returns decoded structure with amount=0)
        - Handles failed transactions (status=0, marked as "failed")
        - Skips transactions with input data (contract calls)
        - All addresses are checksummed using Web3.toChecksumAddress()
    """
    # Extract input data - if present, this is not a simple transfer
    input_data = transaction.get("input", "0x")
    if input_data and input_data != "0x":
        logger.debug(
            f"Transaction {transaction.get('hash', 'unknown')} has input data, "
            f"not a simple ETH transfer"
        )
        return None

    # Extract recipient - if None, this is a contract creation
    to_address = transaction.get("to")
    if to_address is None:
        logger.debug(
            f"Transaction {transaction.get('hash', 'unknown')} has no recipient, "
            f"likely contract creation"
        )
        return None

    # Extract and validate addresses
    from_address = transaction.get("from")
    if not from_address or not to_address:
        logger.warning(
            f"Transaction {transaction.get('hash', 'unknown')} missing from or to address"
        )
        return None

    # Apply checksum to addresses
    try:
        from_checksummed = Web3.to_checksum_address(from_address)
        to_checksummed = Web3.to_checksum_address(to_address)
    except ValueError as e:
        logger.error(
            f"Invalid address in transaction {transaction.get('hash', 'unknown')}: {e}"
        )
        return None

    # Extract value in Wei
    amount_wei = transaction.get("value", 0)

    # Convert to Ether (1 ETH = 10^18 Wei)
    amount_eth = Web3.from_wei(amount_wei, "ether")

    # Extract transaction status from receipt
    status_code = receipt.get("status", 0)
    status = "success" if status_code == 1 else "failed"

    # Extract gas used
    gas_used = receipt.get("gasUsed", 0)

    # Build decoded intent structure
    decoded = {
        "action": "transfer",
        "protocol": "ethereum",
        "from": from_checksummed,
        "to": to_checksummed,
        "amount_wei": amount_wei,
        "amount_eth": float(amount_eth),
        "status": status,
        "gas_used": gas_used,
    }

    logger.debug(
        f"Decoded ETH transfer: {from_checksummed} -> {to_checksummed} "
        f"({amount_eth} ETH, {status})"
    )

    return decoded


def decode_eth_transfers_batch(
    transactions_with_receipts: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Decode multiple ETH transfer transactions in batch.

    Args:
        transactions_with_receipts: List of (transaction, receipt) tuples

    Returns:
        List of decoded intent dictionaries (skips non-transfer transactions)

    Notes:
        - Filters out transactions that are not simple ETH transfers
        - Includes failed transactions in output (marked with status="failed")
    """
    decoded_transfers = []

    for transaction, receipt in transactions_with_receipts:
        decoded = decode_eth_transfer(transaction, receipt)
        if decoded is not None:
            # Add transaction hash and block number for reference
            decoded["tx_hash"] = transaction.get("hash", "unknown")
            decoded["block_number"] = transaction.get("blockNumber", 0)
            decoded_transfers.append(decoded)

    logger.info(
        f"Decoded {len(decoded_transfers)} ETH transfers from "
        f"{len(transactions_with_receipts)} transactions"
    )

    return decoded_transfers
