"""
Decode Uniswap V3 swap transactions.

This module extracts structured intent data from Uniswap V3 swap events
by decoding Swap events from transaction logs with tick-based pricing.

Usage:
    from eth_finetuning.extraction.decoders.uniswap.v3 import decode_uniswap_v3_swaps

    decoded = decode_uniswap_v3_swaps(transaction, receipt, w3)
"""

import logging
from typing import Any

from eth_abi.exceptions import DecodingError
from web3 import Web3
from web3.contract import Contract

from ...core.normalization import normalize_hex_field
from ...core.utils import load_abi

logger = logging.getLogger(__name__)

# Uniswap V3 Swap event signature: Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
UNISWAP_V3_SWAP_SIGNATURE = Web3.keccak(
    text="Swap(address,address,int256,int256,uint160,uint128,int24)"
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
    w3: Web3, pool_address: str, v3_abi: list[dict[str, Any]]
) -> tuple[str | None, str | None]:
    """
    Fetch token0 and token1 addresses from Uniswap V3 pool.

    Args:
        w3: Web3 instance
        pool_address: Pool contract address
        v3_abi: Uniswap V3 pool ABI

    Returns:
        Tuple of (token0_address, token1_address). Returns (None, None) on error.
    """
    try:
        pool_contract: Contract = w3.eth.contract(
            address=Web3.to_checksum_address(pool_address), abi=v3_abi
        )
        token0 = pool_contract.functions.token0().call()
        token1 = pool_contract.functions.token1().call()
        return token0, token1
    except Exception as e:
        logger.warning(f"Failed to fetch pool tokens for {pool_address}: {e}")
        return None, None


def decode_uniswap_v3_swaps(
    transaction: dict[str, Any], receipt: dict[str, Any], w3: Web3
) -> list[dict[str, Any]]:
    """
    Decode Uniswap V3 Swap events from transaction logs.

    Extracts structured intent data for all Uniswap V3 swaps in a transaction.
    V3 uses tick-based concentrated liquidity with signed deltas for amounts.

    Args:
        transaction: Transaction data containing hash, from, to fields
        receipt: Transaction receipt containing logs and status
        w3: Web3 instance for contract interactions (fetching pool/token metadata)

    Returns:
        List of decoded swap dictionaries with structure:
        {
            "action": "swap",
            "protocol": "uniswap_v3",
            "pool_address": checksummed pool contract address,
            "token0": checksummed token0 address,
            "token1": checksummed token1 address,
            "token0_symbol": str (e.g., "USDC", "UNKNOWN" if unavailable),
            "token1_symbol": str (e.g., "WETH", "UNKNOWN" if unavailable),
            "amount0": int (signed delta of token0, negative = sent, positive = received),
            "amount1": int (signed delta of token1, negative = sent, positive = received),
            "sqrt_price_x96": int (price after swap in Q96 format),
            "liquidity": int (liquidity of pool),
            "tick": int (tick after swap),
            "token_in": checksummed address of input token,
            "token_out": checksummed address of output token,
            "token_in_symbol": str,
            "token_out_symbol": str,
            "amount_in": int (absolute value of amount sent),
            "amount_out": int (absolute value of amount received),
            "amount_in_formatted": float | None (amount adjusted for decimals),
            "amount_out_formatted": float | None (amount adjusted for decimals),
            "sender": checksummed address of swap initiator,
            "recipient": checksummed address of swap recipient,
            "status": "success" | "failed",
            "log_index": int
        }
        Returns empty list if no Uniswap V3 swaps found

    Notes:
        - Handles failed transactions (status=0, marked as "failed")
        - Handles tokens without metadata (symbol="UNKNOWN", decimals=None)
        - Decodes signed int256 values for amount deltas
        - Negative amounts indicate tokens sent to pool, positive indicate received
        - All addresses are checksummed using Web3.to_checksum_address()
        - Catches decoding errors for malformed logs

    V3 Swap Direction Logic:
        - amount0 and amount1 are signed integers (int256)
        - Negative value = tokens sent to pool (input)
        - Positive value = tokens received from pool (output)
        - If amount0 < 0 and amount1 > 0: token0 → token1
        - If amount1 < 0 and amount0 > 0: token1 → token0
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
        v3_abi = load_abi("uniswap_v3")
        erc20_abi = load_abi("erc20")
    except FileNotFoundError as e:
        logger.error(f"Required ABI not found: {e}")
        return []

    decoded_swaps = []

    for log in logs:
        # Check if log is a Uniswap V3 Swap event by signature
        topics = log.get("topics", [])
        if not topics or topics[0] != UNISWAP_V3_SWAP_SIGNATURE:
            continue

        try:
            # Decode Swap event
            # Topics: [signature, indexed sender, indexed recipient]
            # Data: [amount0, amount1, sqrtPriceX96, liquidity, tick]
            if len(topics) != 3:
                logger.warning(
                    f"Uniswap V3 Swap event has unexpected number of topics: {len(topics)}"
                )
                continue

            # Extract indexed parameters (sender and recipient addresses)
            sender_address = "0x" + topics[1][-40:]
            recipient_address = "0x" + topics[2][-40:]

            # Extract pool address from log
            pool_address = log.get("address")
            if not pool_address:
                logger.warning("Swap event missing pool address")
                continue

            # Decode data (non-indexed parameters) using normalization utility
            data = log.get("data", "0x")
            if data == "0x" or data == "":
                logger.warning("Swap event has empty data field")
                continue

            # Parse hex data - handles both with and without 0x prefix
            data_bytes = normalize_hex_field(data)

            # amount0: int256 (32 bytes)
            # amount1: int256 (32 bytes)
            # sqrtPriceX96: uint160 (32 bytes, padded)
            # liquidity: uint128 (32 bytes, padded)
            # tick: int24 (32 bytes, padded)
            # Total: 160 bytes
            if len(data_bytes) < 160:
                logger.warning(
                    f"Swap event data too short: {len(data_bytes)} bytes (expected 160)"
                )
                continue

            # Decode signed integers using two's complement
            def decode_int256(data: bytes) -> int:
                """Decode int256 from bytes using two's complement."""
                value = int.from_bytes(data, byteorder="big", signed=False)
                # Check if negative (high bit set)
                if value >= 2**255:
                    value -= 2**256
                return value

            amount0 = decode_int256(data_bytes[0:32])
            amount1 = decode_int256(data_bytes[32:64])
            sqrt_price_x96 = int.from_bytes(data_bytes[64:96], byteorder="big")
            liquidity = int.from_bytes(data_bytes[96:128], byteorder="big")

            # Decode tick (int24 in 32 bytes, sign-extended)
            tick_bytes = data_bytes[128:160]
            tick = int.from_bytes(tick_bytes, byteorder="big", signed=False)
            # Check if negative (bit 23 set in a 24-bit signed integer)
            if tick >= 2**23:
                tick -= 2**24

            # Get pool token addresses
            token0_address, token1_address = _get_pool_tokens(w3, pool_address, v3_abi)
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

            # Determine swap direction from signed amounts
            # Negative = sent to pool (input), Positive = received from pool (output)
            if amount0 < 0 and amount1 > 0:
                # Swapping token0 → token1
                token_in = token0_checksummed
                token_out = token1_checksummed
                token_in_symbol = token0_symbol
                token_out_symbol = token1_symbol
                amount_in = abs(amount0)
                amount_out = abs(amount1)
                decimals_in = token0_decimals
                decimals_out = token1_decimals
            elif amount1 < 0 and amount0 > 0:
                # Swapping token1 → token0
                token_in = token1_checksummed
                token_out = token0_checksummed
                token_in_symbol = token1_symbol
                token_out_symbol = token0_symbol
                amount_in = abs(amount1)
                amount_out = abs(amount0)
                decimals_in = token1_decimals
                decimals_out = token0_decimals
            else:
                # Unusual case - both negative or both positive
                logger.warning(
                    f"Unusual swap amounts in {pool_address}: "
                    f"amount0={amount0}, amount1={amount1}"
                )
                # Default to token0 as input
                token_in = token0_checksummed
                token_out = token1_checksummed
                token_in_symbol = token0_symbol
                token_out_symbol = token1_symbol
                amount_in = abs(amount0)
                amount_out = abs(amount1)
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
                "protocol": "uniswap_v3",
                "pool_address": pool_checksummed,
                "token0": token0_checksummed,
                "token1": token1_checksummed,
                "token0_symbol": token0_symbol,
                "token1_symbol": token1_symbol,
                "amount0": amount0,
                "amount1": amount1,
                "sqrt_price_x96": sqrt_price_x96,
                "liquidity": liquidity,
                "tick": tick,
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
                f"Decoded Uniswap V3 swap: {token_in_symbol} → {token_out_symbol} "
                f"in pool {pool_checksummed} at tick {tick}"
            )

        except DecodingError as e:
            logger.error(f"Failed to decode Uniswap V3 Swap event: {e}")
            logger.error(f"Log data: {log}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error decoding Uniswap V3 Swap event: {e}")
            logger.error(f"Log data: {log}")
            continue

    return decoded_swaps
