"""
Decode Uniswap V2 swap transactions.

This module extracts structured intent data from Uniswap V2 swap events
by decoding Swap events from transaction logs.

Usage:
    from eth_finetuning.extraction.decoders.uniswap.v2 import decode_uniswap_v2_swaps

    decoded = decode_uniswap_v2_swaps(transaction, receipt, w3)
"""

import logging
from typing import Any

from eth_abi.exceptions import DecodingError
from web3 import Web3
from web3.contract import Contract

from ...core.utils import load_abi

logger = logging.getLogger(__name__)

# Uniswap V2 Swap event signature: Swap(address indexed sender, uint256 amount0In, uint256 amount1In, uint256 amount0Out, uint256 amount1Out, address indexed to)
UNISWAP_V2_SWAP_SIGNATURE = Web3.keccak(
    text="Swap(address,uint256,uint256,uint256,uint256,address)"
).hex()


def _get_token_metadata(
    w3: Web3, token_address: str, erc20_abi: list[dict[str, Any]]
) -> tuple[str, int | None]:
    """
    Fetch token symbol and decimals from contract.

    Args:
        w3: Web3 instance
        token_address: Token contract address
        erc20_abi: ERC-20 ABI for contract interaction

    Returns:
        Tuple of (symbol, decimals). Symbol defaults to "UNKNOWN" on error,
        decimals defaults to None on error.
    """
    try:
        token_contract = w3.eth.contract(
            address=Web3.to_checksum_address(token_address), abi=erc20_abi
        )
        try:
            symbol = token_contract.functions.symbol().call()
        except Exception:
            symbol = "UNKNOWN"

        try:
            decimals = token_contract.functions.decimals().call()
        except Exception:
            decimals = None

        return symbol, decimals
    except Exception as e:
        logger.warning(f"Failed to fetch token metadata for {token_address}: {e}")
        return "UNKNOWN", None


def _get_pool_tokens(
    w3: Web3, pool_address: str, v2_abi: list[dict[str, Any]]
) -> tuple[str | None, str | None]:
    """
    Fetch token0 and token1 addresses from Uniswap V2 pool.

    Args:
        w3: Web3 instance
        pool_address: Pool contract address
        v2_abi: Uniswap V2 pool ABI

    Returns:
        Tuple of (token0_address, token1_address). Returns (None, None) on error.
    """
    try:
        pool_contract: Contract = w3.eth.contract(
            address=Web3.to_checksum_address(pool_address), abi=v2_abi
        )
        token0 = pool_contract.functions.token0().call()
        token1 = pool_contract.functions.token1().call()
        return token0, token1
    except Exception as e:
        logger.warning(f"Failed to fetch pool tokens for {pool_address}: {e}")
        return None, None


def decode_uniswap_v2_swaps(
    transaction: dict[str, Any], receipt: dict[str, Any], w3: Web3
) -> list[dict[str, Any]]:
    """
    Decode Uniswap V2 Swap events from transaction logs.

    Extracts structured intent data for all Uniswap V2 swaps in a transaction.
    Multiple swaps can occur in a single transaction (multi-hop routing).

    Args:
        transaction: Transaction data containing hash, from, to fields
        receipt: Transaction receipt containing logs and status
        w3: Web3 instance for contract interactions (fetching pool/token metadata)

    Returns:
        List of decoded swap dictionaries with structure:
        {
            "action": "swap",
            "protocol": "uniswap_v2",
            "pool_address": checksummed pool contract address,
            "token0": checksummed token0 address,
            "token1": checksummed token1 address,
            "token0_symbol": str (e.g., "USDC", "UNKNOWN" if unavailable),
            "token1_symbol": str (e.g., "WETH", "UNKNOWN" if unavailable),
            "amount0_in": int (amount of token0 sent to pool),
            "amount1_in": int (amount of token1 sent to pool),
            "amount0_out": int (amount of token0 received from pool),
            "amount1_out": int (amount of token1 received from pool),
            "token_in": checksummed address of input token,
            "token_out": checksummed address of output token,
            "token_in_symbol": str,
            "token_out_symbol": str,
            "amount_in": int (raw amount in),
            "amount_out": int (raw amount out),
            "amount_in_formatted": float | None (amount adjusted for decimals),
            "amount_out_formatted": float | None (amount adjusted for decimals),
            "sender": checksummed address of swap initiator,
            "recipient": checksummed address of swap recipient,
            "status": "success" | "failed",
            "log_index": int
        }
        Returns empty list if no Uniswap V2 swaps found

    Notes:
        - Handles failed transactions (status=0, marked as "failed")
        - Handles tokens without metadata (symbol="UNKNOWN", decimals=None)
        - Determines swap direction from amount0In/Out and amount1In/Out
        - For multi-hop swaps, each intermediate swap is returned separately
        - All addresses are checksummed using Web3.to_checksum_address()
        - Catches decoding errors for malformed logs

    Swap Direction Logic:
        - If amount0In > 0 and amount1Out > 0: token0 → token1
        - If amount1In > 0 and amount0Out > 0: token1 → token0
        - Sets token_in/token_out and amount_in/amount_out accordingly
    """
    logs = receipt.get("logs", [])
    if not logs:
        logger.debug(f"Transaction {transaction.get('hash', 'unknown')} has no logs")
        return []

    # Extract transaction status
    status_code = receipt.get("status", 0)
    status = "success" if status_code == 1 else "failed"

    # Load ABIs
    try:
        v2_abi = load_abi("uniswap_v2")
        erc20_abi = load_abi("erc20")
    except FileNotFoundError as e:
        logger.error(f"Required ABI not found: {e}")
        return []

    decoded_swaps = []

    for log in logs:
        # Check if log is a Uniswap V2 Swap event by signature
        topics = log.get("topics", [])
        if not topics or topics[0] != UNISWAP_V2_SWAP_SIGNATURE:
            continue

        try:
            # Decode Swap event
            # Topics: [signature, indexed sender, indexed to]
            # Data: [amount0In, amount1In, amount0Out, amount1Out]
            if len(topics) != 3:
                logger.warning(
                    f"Uniswap V2 Swap event has unexpected number of topics: {len(topics)}"
                )
                continue

            # Extract indexed parameters (sender and to addresses)
            sender_address = "0x" + topics[1][-40:]
            recipient_address = "0x" + topics[2][-40:]

            # Extract pool address from log
            pool_address = log.get("address")
            if not pool_address:
                logger.warning("Swap event missing pool address")
                continue

            # Decode data (non-indexed parameters)
            data = log.get("data", "0x")
            if data == "0x":
                logger.warning("Swap event has empty data field")
                continue

            # Remove 0x prefix and decode
            data_bytes = bytes.fromhex(data[2:])

            # Each uint256 is 32 bytes
            if len(data_bytes) < 128:  # 4 * 32 bytes
                logger.warning(
                    f"Swap event data too short: {len(data_bytes)} bytes (expected 128)"
                )
                continue

            amount0_in = int.from_bytes(data_bytes[0:32], byteorder="big")
            amount1_in = int.from_bytes(data_bytes[32:64], byteorder="big")
            amount0_out = int.from_bytes(data_bytes[64:96], byteorder="big")
            amount1_out = int.from_bytes(data_bytes[96:128], byteorder="big")

            # Get pool token addresses
            token0_address, token1_address = _get_pool_tokens(w3, pool_address, v2_abi)
            if not token0_address or not token1_address:
                logger.warning(f"Failed to get token addresses for pool {pool_address}")
                # Continue anyway with partial data
                token0_address = "0x0000000000000000000000000000000000000000"
                token1_address = "0x0000000000000000000000000000000000000000"

            # Apply checksum to addresses
            pool_checksummed = Web3.to_checksum_address(pool_address)
            sender_checksummed = Web3.to_checksum_address(sender_address)
            recipient_checksummed = Web3.to_checksum_address(recipient_address)
            token0_checksummed = Web3.to_checksum_address(token0_address)
            token1_checksummed = Web3.to_checksum_address(token1_address)

            # Get token metadata
            token0_symbol, token0_decimals = _get_token_metadata(
                w3, token0_address, erc20_abi
            )
            token1_symbol, token1_decimals = _get_token_metadata(
                w3, token1_address, erc20_abi
            )

            # Determine swap direction
            # If amount0In > 0, then token0 is the input
            # If amount1Out > 0, then token1 is the output
            if amount0_in > 0 and amount1_out > 0:
                # Swapping token0 → token1
                token_in = token0_checksummed
                token_out = token1_checksummed
                token_in_symbol = token0_symbol
                token_out_symbol = token1_symbol
                amount_in = amount0_in
                amount_out = amount1_out
                decimals_in = token0_decimals
                decimals_out = token1_decimals
            elif amount1_in > 0 and amount0_out > 0:
                # Swapping token1 → token0
                token_in = token1_checksummed
                token_out = token0_checksummed
                token_in_symbol = token1_symbol
                token_out_symbol = token0_symbol
                amount_in = amount1_in
                amount_out = amount0_out
                decimals_in = token1_decimals
                decimals_out = token0_decimals
            else:
                # Unusual case - log it and skip direction determination
                logger.warning(
                    f"Unusual swap amounts in {pool_address}: "
                    f"amount0In={amount0_in}, amount1In={amount1_in}, "
                    f"amount0Out={amount0_out}, amount1Out={amount1_out}"
                )
                token_in = token0_checksummed
                token_out = token1_checksummed
                token_in_symbol = token0_symbol
                token_out_symbol = token1_symbol
                amount_in = amount0_in or amount1_in
                amount_out = amount0_out or amount1_out
                decimals_in = token0_decimals
                decimals_out = token1_decimals

            # Format amounts with decimals if available
            amount_in_formatted = None
            if decimals_in is not None:
                amount_in_formatted = amount_in / (10**decimals_in)

            amount_out_formatted = None
            if decimals_out is not None:
                amount_out_formatted = amount_out / (10**decimals_out)

            decoded_swap = {
                "action": "swap",
                "protocol": "uniswap_v2",
                "pool_address": pool_checksummed,
                "token0": token0_checksummed,
                "token1": token1_checksummed,
                "token0_symbol": token0_symbol,
                "token1_symbol": token1_symbol,
                "amount0_in": amount0_in,
                "amount1_in": amount1_in,
                "amount0_out": amount0_out,
                "amount1_out": amount1_out,
                "token_in": token_in,
                "token_out": token_out,
                "token_in_symbol": token_in_symbol,
                "token_out_symbol": token_out_symbol,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "amount_in_formatted": amount_in_formatted,
                "amount_out_formatted": amount_out_formatted,
                "sender": sender_checksummed,
                "recipient": recipient_checksummed,
                "status": status,
                "log_index": log.get("logIndex", 0),
            }

            decoded_swaps.append(decoded_swap)
            logger.debug(
                f"Decoded Uniswap V2 swap: {token_in_symbol} → {token_out_symbol} "
                f"in pool {pool_checksummed}"
            )

        except DecodingError as e:
            logger.error(f"Failed to decode Uniswap V2 Swap event: {e}")
            logger.error(f"Log data: {log}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error decoding Uniswap V2 Swap event: {e}")
            logger.error(f"Log data: {log}")
            continue

    return decoded_swaps
