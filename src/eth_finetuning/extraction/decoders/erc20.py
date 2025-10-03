"""
Decode ERC-20 token transfer transactions.

This module extracts structured intent data from ERC-20 token transfers
by decoding Transfer events from transaction logs.

Usage:
    from eth_finetuning.extraction.decoders.erc20 import decode_erc20_transfers

    decoded = decode_erc20_transfers(transaction, receipt, w3)
"""

import logging
from typing import Any

from eth_abi.exceptions import DecodingError
from web3 import Web3

from ..core.utils import load_abi

logger = logging.getLogger(__name__)

# ERC-20 Transfer event signature: Transfer(address indexed from, address indexed to, uint256 value)
TRANSFER_EVENT_SIGNATURE = (
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
)


def decode_erc20_transfers(
    transaction: dict[str, Any], receipt: dict[str, Any], w3: Web3
) -> list[dict[str, Any]]:
    """
    Decode ERC-20 Transfer events from transaction logs.

    Extracts structured intent data for all ERC-20 token transfers in a transaction.
    Multiple transfers can occur in a single transaction (e.g., multi-send, DEX swaps).

    Args:
        transaction: Transaction data containing hash, from, to fields
        receipt: Transaction receipt containing logs and status
        w3: Web3 instance for contract interactions (fetching token metadata)

    Returns:
        List of decoded transfer dictionaries with structure:
        {
            "action": "transfer",
            "protocol": "erc20",
            "token_address": checksummed token contract address,
            "token_symbol": str (e.g., "USDC", "UNKNOWN" if unavailable),
            "token_decimals": int (e.g., 6, 18, or None if unavailable),
            "from": checksummed sender address,
            "to": checksummed recipient address,
            "amount": int (raw amount in token's smallest unit),
            "amount_formatted": float (amount adjusted for decimals, or None),
            "status": "success" | "failed",
            "log_index": int
        }
        Returns empty list if no ERC-20 transfers found

    Notes:
        - Handles failed transactions (status=0, marked as "failed")
        - Handles tokens without name/symbol/decimals (non-standard implementations)
        - Catches decoding errors for malformed logs
        - All addresses are checksummed using Web3.to_checksum_address()
    """
    logs = receipt.get("logs", [])
    if not logs:
        logger.debug(f"Transaction {transaction.get('hash', 'unknown')} has no logs")
        return []

    # Extract transaction status
    status_code = receipt.get("status", 0)
    status = "success" if status_code == 1 else "failed"

    # Load ERC-20 ABI for contract interaction
    try:
        erc20_abi = load_abi("erc20")
    except FileNotFoundError:
        logger.error("ERC-20 ABI not found. Cannot decode transfers.")
        return []

    decoded_transfers = []

    for log in logs:
        # Check if log is a Transfer event by signature
        topics = log.get("topics", [])
        if not topics or topics[0] != TRANSFER_EVENT_SIGNATURE:
            continue

        try:
            # Decode Transfer event
            # Topics: [signature, indexed from, indexed to]
            # Data: [value]
            if len(topics) != 3:
                logger.warning(
                    f"Transfer event has unexpected number of topics: {len(topics)}"
                )
                continue

            # Extract addresses from indexed topics (remove 0x prefix and leading zeros)
            from_address = Web3.to_checksum_address(
                "0x" + topics[1][2:].lstrip("0")[-40:]
            )
            to_address = Web3.to_checksum_address(
                "0x" + topics[2][2:].lstrip("0")[-40:]
            )

            # Decode amount from data field
            data = log.get("data", "0x")
            if data == "0x" or len(data) < 66:  # 0x + 64 hex chars for uint256
                logger.warning(f"Transfer event has invalid data field: {data}")
                continue

            # Convert hex data to integer (uint256)
            amount = int(data, 16)

            # Get token contract address
            token_address = Web3.to_checksum_address(log.get("address"))

            # Attempt to fetch token metadata (symbol, decimals)
            token_symbol, token_decimals = _get_token_metadata(
                w3, token_address, erc20_abi
            )

            # Format amount with decimals if available
            amount_formatted = None
            if token_decimals is not None:
                amount_formatted = amount / (10**token_decimals)

            # Build decoded transfer structure
            decoded = {
                "action": "transfer",
                "protocol": "erc20",
                "token_address": token_address,
                "token_symbol": token_symbol,
                "token_decimals": token_decimals,
                "from": from_address,
                "to": to_address,
                "amount": amount,
                "amount_formatted": amount_formatted,
                "status": status,
                "log_index": log.get("logIndex", 0),
            }

            decoded_transfers.append(decoded)

            logger.debug(
                f"Decoded ERC-20 transfer: {from_address} -> {to_address} "
                f"({amount_formatted or amount} {token_symbol})"
            )

        except (ValueError, DecodingError) as e:
            logger.warning(
                f"Failed to decode Transfer event in transaction "
                f"{transaction.get('hash', 'unknown')}: {e}"
            )
            continue

    if decoded_transfers:
        logger.info(
            f"Decoded {len(decoded_transfers)} ERC-20 transfers from transaction "
            f"{transaction.get('hash', 'unknown')}"
        )

    return decoded_transfers


def _get_token_metadata(
    w3: Web3, token_address: str, erc20_abi: list[dict[str, Any]]
) -> tuple[str, int | None]:
    """
    Fetch token symbol and decimals from contract.

    Attempts to call symbol() and decimals() functions on token contract.
    Returns default values if contract doesn't implement these functions
    (non-standard ERC-20 implementations).

    Args:
        w3: Web3 instance
        token_address: Checksummed token contract address
        erc20_abi: ERC-20 ABI definition

    Returns:
        Tuple of (symbol, decimals):
        - symbol: Token symbol string, or "UNKNOWN" if unavailable
        - decimals: Number of decimals (int), or None if unavailable

    Notes:
        - Catches all exceptions to handle non-standard tokens gracefully
        - Does not retry on failure (metadata is optional)
    """
    try:
        # Create contract instance
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(token_address), abi=erc20_abi
        )

        # Fetch symbol
        try:
            symbol = contract.functions.symbol().call()
        except Exception as e:
            logger.debug(f"Failed to fetch symbol for {token_address}: {e}")
            symbol = "UNKNOWN"

        # Fetch decimals
        try:
            decimals = contract.functions.decimals().call()
        except Exception as e:
            logger.debug(f"Failed to fetch decimals for {token_address}: {e}")
            decimals = None

        return symbol, decimals

    except Exception as e:
        logger.warning(f"Failed to create contract for {token_address}: {e}")
        return "UNKNOWN", None


def decode_erc20_transfers_batch(
    transactions_with_receipts: list[tuple[dict[str, Any], dict[str, Any]]], w3: Web3
) -> list[dict[str, Any]]:
    """
    Decode ERC-20 transfers from multiple transactions in batch.

    Args:
        transactions_with_receipts: List of (transaction, receipt) tuples
        w3: Web3 instance for contract interactions

    Returns:
        List of all decoded ERC-20 transfer dictionaries across all transactions

    Notes:
        - Flattens results from multiple transactions into single list
        - Each decoded transfer includes tx_hash and block_number for reference
        - Includes transfers from failed transactions (marked with status="failed")
    """
    all_transfers = []

    for transaction, receipt in transactions_with_receipts:
        decoded_transfers = decode_erc20_transfers(transaction, receipt, w3)

        # Add transaction hash and block number to each transfer
        for transfer in decoded_transfers:
            transfer["tx_hash"] = transaction.get("hash", "unknown")
            transfer["block_number"] = transaction.get("blockNumber", 0)

        all_transfers.extend(decoded_transfers)

    logger.info(
        f"Decoded {len(all_transfers)} ERC-20 transfers from "
        f"{len(transactions_with_receipts)} transactions"
    )

    return all_transfers
