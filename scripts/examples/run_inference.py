#!/usr/bin/env python3
"""
Run inference on Ethereum transactions using fine-tuned model.

This script demonstrates how to load a fine-tuned model with adapter and
run inference on transaction data to extract structured intents.

Usage:
    # Inference on a single transaction hash (fetches from RPC)
    python scripts/examples/run_inference.py \\
        --model models/fine-tuned/eth-intent-extractor-v1 \\
        --tx-hash 0xabcdef1234567890... \\
        --rpc-url https://mainnet.infura.io/v3/YOUR_KEY \\
        --output outputs/predictions/single_inference.json

    # Inference on pre-fetched transaction data (JSON file)
    python scripts/examples/run_inference.py \\
        --model models/fine-tuned/eth-intent-extractor-v1 \\
        --input-file data/raw/transaction.json \\
        --output outputs/predictions/inference.json

Educational Notes:
    - Model is loaded in 4-bit quantized format for memory efficiency
    - Inference runs in torch.no_grad() context to disable gradients
    - Low temperature (0.1) produces deterministic, focused outputs
    - JSON parsing includes error handling for malformed model outputs
    - Compatible with consumer GPUs (RTX 3060, ~4GB VRAM usage)
"""

import json
import logging
import sys
from pathlib import Path

import click
import torch

from eth_finetuning.dataset.templates import DEFAULT_INSTRUCTION
from eth_finetuning.evaluation.evaluator import (
    load_model_for_evaluation,
    parse_json_output,
    run_inference,
)
from eth_finetuning.extraction.core.fetcher import fetch_transactions_batch
from eth_finetuning.extraction.core.utils import Web3ConnectionManager, setup_logging
from eth_finetuning.extraction.decoders.eth import decode_eth_transfer
from eth_finetuning.extraction.decoders.erc20 import decode_erc20_transfers
from eth_finetuning.extraction.decoders.uniswap import (
    decode_uniswap_v2_swaps,
    decode_uniswap_v3_swaps,
)

logger = logging.getLogger(__name__)


def fetch_and_decode_transaction(tx_hash: str, rpc_url: str) -> dict | None:
    """
    Fetch and decode a single transaction from Ethereum RPC.

    This function demonstrates the full data extraction pipeline:
    1. Connect to Ethereum RPC endpoint
    2. Fetch transaction data and receipt
    3. Decode transaction using protocol-specific decoders

    Args:
        tx_hash: Transaction hash (with or without 0x prefix)
        rpc_url: Ethereum RPC endpoint URL

    Returns:
        Decoded transaction dictionary, or None if transaction not found

    Raises:
        ConnectionError: If RPC connection fails
        ValueError: If transaction hash is invalid
    """
    logger.info(f"Fetching transaction {tx_hash} from RPC...")

    # Ensure tx_hash has 0x prefix
    if not tx_hash.startswith("0x"):
        tx_hash = f"0x{tx_hash}"

    try:
        # Connect to Web3
        w3_manager = Web3ConnectionManager(rpc_url)

        # Fetch transaction data
        transactions = fetch_transactions_batch([tx_hash], w3_manager)

        if not transactions or len(transactions) == 0:
            logger.error(f"Transaction {tx_hash} not found")
            return None

        tx_data = transactions[0]

        # Extract transaction and receipt
        tx = tx_data["transaction"]
        receipt = tx_data["receipt"]

        logger.info(
            f"Transaction fetched successfully. Status: {receipt.get('status')}"
        )

        # Decode transaction using available decoders
        decoded = decode_transaction_with_all_decoders(tx, receipt, w3_manager)

        if not decoded:
            logger.warning("No decoder matched this transaction")
            # Return basic structure for unsupported transactions
            return {
                "tx_hash": tx_hash,
                "protocol": "unknown",
                "action": "unknown",
                "status": "success" if receipt.get("status") == 1 else "failed",
            }

        # Return the first successful decode (or prioritize by type)
        return decoded[0] if decoded else None

    except ConnectionError as e:
        logger.error(f"RPC connection failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch/decode transaction: {e}")
        raise


def decode_transaction_with_all_decoders(
    tx: dict, receipt: dict, w3_manager: Web3ConnectionManager
) -> list[dict]:
    """
    Run all available decoders on a transaction.

    Tries each decoder in sequence and collects successful decodes.

    Args:
        tx: Transaction data dictionary
        receipt: Transaction receipt dictionary
        w3_manager: Web3 connection manager

    Returns:
        List of decoded transaction dictionaries
    """
    decoded = []

    # Try ETH transfer decoder
    try:
        eth_result = decode_eth_transfer(tx, receipt)
        if eth_result:
            decoded.append(eth_result)
            logger.debug("ETH transfer decoded")
    except Exception as e:
        logger.debug(f"ETH decoder failed: {e}")

    # Try ERC-20 decoder
    try:
        erc20_results = decode_erc20_transfers(tx, receipt, w3_manager.w3)
        if erc20_results:
            decoded.extend(erc20_results)
            logger.debug(f"ERC-20 transfers decoded: {len(erc20_results)}")
    except Exception as e:
        logger.debug(f"ERC-20 decoder failed: {e}")

    # Try Uniswap V2 decoder
    try:
        uniswap_v2_results = decode_uniswap_v2_swaps(tx, receipt, w3_manager.w3)
        if uniswap_v2_results:
            decoded.extend(uniswap_v2_results)
            logger.debug(f"Uniswap V2 swaps decoded: {len(uniswap_v2_results)}")
    except Exception as e:
        logger.debug(f"Uniswap V2 decoder failed: {e}")

    # Try Uniswap V3 decoder
    try:
        uniswap_v3_results = decode_uniswap_v3_swaps(tx, receipt, w3_manager.w3)
        if uniswap_v3_results:
            decoded.extend(uniswap_v3_results)
            logger.debug(f"Uniswap V3 swaps decoded: {len(uniswap_v3_results)}")
    except Exception as e:
        logger.debug(f"Uniswap V3 decoder failed: {e}")

    return decoded


def format_prompt(decoded_tx: dict, instruction: str = DEFAULT_INSTRUCTION) -> str:
    """
    Format decoded transaction as a prompt for the model.

    This replicates the prompt format used during training.

    Args:
        decoded_tx: Decoded transaction dictionary
        instruction: Instruction text (default: from templates.py)

    Returns:
        Formatted prompt string
    """
    # Format transaction data as JSON
    tx_json = json.dumps(decoded_tx, indent=None, separators=(",", ":"))

    # Create prompt in Alpaca format (instruction + input)
    prompt = f"{instruction}\n\nTransaction data:\n{tx_json}\n\nExtracted intent:"

    return prompt


def load_transaction_from_file(file_path: Path) -> dict:
    """
    Load transaction data from JSON file.

    Expected format:
        - Single transaction: {tx_hash, transaction, receipt, ...}
        - Already decoded: {tx_hash, protocol, action, ...}

    Args:
        file_path: Path to JSON file

    Returns:
        Transaction data dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or missing required fields
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Handle both raw and decoded transaction formats
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Input file contains empty list")
        data = data[0]  # Take first transaction

    # Check if already decoded (has protocol field)
    if "protocol" in data:
        return data

    # If raw transaction, need to decode
    if "transaction" in data and "receipt" in data:
        # This is raw format from fetch_transactions.py
        # For simplicity, require --rpc-url for decoding
        raise ValueError(
            "Raw transaction format requires --rpc-url for decoding. "
            "Use --tx-hash instead of --input-file, or provide pre-decoded data."
        )

    raise ValueError(
        "Invalid transaction format. Expected decoded transaction with 'protocol' field."
    )


def save_result(result: dict, output_path: Path, pretty: bool = True) -> None:
    """
    Save inference result to JSON file.

    Args:
        result: Result dictionary containing decoded_tx and extracted_intent
        output_path: Path to output JSON file
        pretty: Whether to format JSON with indentation
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        if pretty:
            json.dump(result, f, indent=2)
        else:
            json.dump(result, f)

    logger.info(f"Result saved to {output_path}")


@click.command()
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to fine-tuned model directory (with adapter)",
)
@click.option(
    "--tx-hash",
    type=str,
    default=None,
    help="Transaction hash to analyze (requires --rpc-url)",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to JSON file with decoded transaction data",
)
@click.option(
    "--rpc-url",
    type=str,
    default=None,
    help="Ethereum RPC endpoint URL (required if using --tx-hash)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSON file path for inference result",
)
@click.option(
    "--max-tokens",
    type=int,
    default=512,
    help="Maximum tokens to generate (default: 512)",
)
@click.option(
    "--temperature",
    type=float,
    default=0.1,
    help="Sampling temperature (default: 0.1 for deterministic output)",
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
    model: Path,
    tx_hash: str | None,
    input_file: Path | None,
    rpc_url: str | None,
    output: Path,
    max_tokens: int,
    temperature: float,
    log_level: str,
) -> None:
    """
    Run inference on Ethereum transaction using fine-tuned model.

    This script demonstrates the complete inference workflow:
    1. Load fine-tuned model with adapter
    2. Fetch and decode transaction (if tx_hash provided)
    3. Format transaction as prompt
    4. Generate intent prediction
    5. Parse and save result

    Either --tx-hash or --input-file must be provided.
    If using --tx-hash, --rpc-url is also required.

    Examples:
        # Analyze live transaction
        python run_inference.py --model models/fine-tuned/v1 \\
            --tx-hash 0xabc... --rpc-url https://mainnet.infura.io/v3/KEY \\
            --output result.json

        # Analyze from file
        python run_inference.py --model models/fine-tuned/v1 \\
            --input-file data/decoded_tx.json --output result.json
    """
    # Setup logging
    setup_logging(log_level)

    # Validate input arguments
    if not tx_hash and not input_file:
        logger.error("Either --tx-hash or --input-file must be provided")
        sys.exit(1)

    if tx_hash and not rpc_url:
        logger.error("--rpc-url is required when using --tx-hash")
        sys.exit(1)

    if tx_hash and input_file:
        logger.warning("Both --tx-hash and --input-file provided, using --tx-hash")

    try:
        # Step 1: Load model
        logger.info(f"Loading model from {model}...")
        model_obj, tokenizer = load_model_for_evaluation(
            model_path=model,
            device_map="auto",
            load_in_4bit=True,
        )
        logger.info(f"Model loaded successfully on device: {model_obj.device}")

        # Step 2: Get transaction data
        if tx_hash:
            logger.info("Fetching transaction from RPC...")
            decoded_tx = fetch_and_decode_transaction(tx_hash, rpc_url)
            if not decoded_tx:
                logger.error("Failed to fetch/decode transaction")
                sys.exit(1)
        else:
            logger.info(f"Loading transaction from {input_file}...")
            decoded_tx = load_transaction_from_file(input_file)

        logger.info(f"Transaction data: {decoded_tx.get('tx_hash', 'unknown')}")
        logger.info(f"Protocol: {decoded_tx.get('protocol', 'unknown')}")
        logger.info(f"Action: {decoded_tx.get('action', 'unknown')}")

        # Step 3: Format prompt
        logger.info("Formatting prompt...")
        prompt = format_prompt(decoded_tx)
        logger.debug(f"Prompt: {prompt[:200]}...")

        # Step 4: Run inference
        logger.info("Running inference...")
        with torch.no_grad():
            generated_text = run_inference(
                model=model_obj,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        logger.info(f"Generated text: {generated_text[:200]}...")

        # Step 5: Parse output
        logger.info("Parsing generated output...")
        extracted_intent = parse_json_output(generated_text)

        if extracted_intent is None:
            logger.warning("Failed to parse JSON output, saving raw text")
            extracted_intent = {
                "error": "Failed to parse JSON",
                "raw_output": generated_text,
            }

        # Step 6: Save result
        result = {
            "decoded_transaction": decoded_tx,
            "extracted_intent": extracted_intent,
            "model_output_raw": generated_text,
        }

        save_result(result, output)

        # Print summary to stdout
        print("\n" + "=" * 60)
        print("INFERENCE RESULT")
        print("=" * 60)
        print(f"Transaction: {decoded_tx.get('tx_hash', 'unknown')}")
        print(f"Protocol: {decoded_tx.get('protocol', 'unknown')}")
        print(f"Action: {decoded_tx.get('action', 'unknown')}")
        print("\nExtracted Intent:")
        print(json.dumps(extracted_intent, indent=2))
        print("=" * 60)
        print(f"\nFull result saved to: {output}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
