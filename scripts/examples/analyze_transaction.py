#!/usr/bin/env python3
"""
Complete end-to-end transaction analysis pipeline demonstration.

This script demonstrates the full workflow from raw transaction hash to
extracted intent using the fine-tuned model:

    1. Fetch transaction from Ethereum RPC
    2. Decode using protocol-specific decoders
    3. Format as model prompt
    4. Run inference to extract structured intent
    5. Display results with detailed breakdown

Usage:
    python scripts/examples/analyze_transaction.py \\
        --tx-hash 0xabcdef1234567890... \\
        --rpc-url https://mainnet.infura.io/v3/YOUR_KEY \\
        --model models/fine-tuned/eth-intent-extractor-v1 \\
        --output outputs/predictions/analysis.json

Educational Purpose:
    This script is designed as a learning tool to show all pipeline steps
    with clear logging and inline comments. It demonstrates:
    - RPC connection and retry patterns
    - Transaction decoding strategies
    - Prompt engineering for instruction-tuned models
    - Inference best practices (torch.no_grad, low temperature)
    - Error handling for production scenarios

    Each step includes detailed logging so learners can understand what's
    happening at each stage of the pipeline.
"""

import json
import logging
import sys
import time
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


class TransactionAnalyzer:
    """
    Complete transaction analysis pipeline.

    This class encapsulates the full workflow from raw transaction hash
    to extracted structured intent. It's designed to be educational,
    with clear separation of concerns and extensive logging.

    Attributes:
        rpc_url: Ethereum RPC endpoint URL
        model_path: Path to fine-tuned model directory
        w3_manager: Web3 connection manager (initialized lazily)
        model: Loaded model (initialized lazily)
        tokenizer: Loaded tokenizer (initialized lazily)
    """

    def __init__(self, rpc_url: str, model_path: Path):
        """
        Initialize analyzer with RPC and model paths.

        Args:
            rpc_url: Ethereum RPC endpoint URL
            model_path: Path to fine-tuned model directory
        """
        self.rpc_url = rpc_url
        self.model_path = model_path
        self.w3_manager = None
        self.model = None
        self.tokenizer = None

    def initialize(self) -> None:
        """
        Initialize RPC connection and load model.

        This is separated from __init__ to allow for better error handling
        and progress reporting during setup.

        Raises:
            ConnectionError: If RPC connection fails
            RuntimeError: If model loading fails
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING TRANSACTION ANALYZER")
        logger.info("=" * 60)

        # Step 1: Connect to Ethereum RPC
        logger.info(f"\n[1/2] Connecting to Ethereum RPC: {self.rpc_url}")
        try:
            self.w3_manager = Web3ConnectionManager(self.rpc_url)
            block_number = self.w3_manager.w3.eth.block_number
            logger.info(f"✓ Connected successfully. Current block: {block_number}")
        except Exception as e:
            logger.error(f"✗ RPC connection failed: {e}")
            raise ConnectionError(f"Failed to connect to RPC: {e}") from e

        # Step 2: Load fine-tuned model
        logger.info(f"\n[2/2] Loading model from: {self.model_path}")
        logger.info(
            "This may take 30-60 seconds for first load (downloading base model)..."
        )
        try:
            start_time = time.time()
            self.model, self.tokenizer = load_model_for_evaluation(
                model_path=self.model_path,
                device_map="auto",
                load_in_4bit=True,
            )
            load_time = time.time() - start_time
            logger.info(f"✓ Model loaded successfully in {load_time:.1f}s")
            logger.info(f"  Device: {self.model.device}")
            logger.info(f"  Memory: ~4GB VRAM (4-bit quantized)")
        except Exception as e:
            logger.error(f"✗ Model loading failed: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e

        logger.info("\n✓ Initialization complete!\n")

    def fetch_transaction(self, tx_hash: str) -> dict:
        """
        Fetch transaction data from Ethereum RPC.

        Step 1 of pipeline: Connect to Ethereum node and retrieve raw
        transaction data and receipt.

        Args:
            tx_hash: Transaction hash (with or without 0x prefix)

        Returns:
            Dictionary with 'transaction' and 'receipt' keys

        Raises:
            ValueError: If transaction not found or invalid hash
            ConnectionError: If RPC request fails
        """
        logger.info("=" * 60)
        logger.info("STEP 1: FETCH TRANSACTION FROM ETHEREUM")
        logger.info("=" * 60)

        # Normalize hash format
        if not tx_hash.startswith("0x"):
            tx_hash = f"0x{tx_hash}"

        logger.info(f"Transaction hash: {tx_hash}")

        try:
            # Fetch using batch fetcher (handles retries automatically)
            logger.info("Fetching transaction data and receipt...")
            transactions = fetch_transactions_batch([tx_hash], self.w3_manager)

            if not transactions or len(transactions) == 0:
                raise ValueError(f"Transaction {tx_hash} not found")

            tx_data = transactions[0]
            logger.info(f"✓ Transaction fetched successfully")

            # Log transaction details
            tx = tx_data["transaction"]
            receipt = tx_data["receipt"]

            logger.info(f"\nTransaction Details:")
            logger.info(f"  Block: {tx.get('blockNumber', 'pending')}")
            logger.info(f"  From: {tx.get('from', 'unknown')}")
            logger.info(f"  To: {tx.get('to', 'contract creation')}")
            logger.info(f"  Value: {tx.get('value', 0)} Wei")
            logger.info(f"  Gas Used: {receipt.get('gasUsed', 'unknown')}")
            logger.info(
                f"  Status: {'Success' if receipt.get('status') == 1 else 'Failed'}"
            )
            logger.info(f"  Logs: {len(receipt.get('logs', []))} events")

            return tx_data

        except Exception as e:
            logger.error(f"✗ Failed to fetch transaction: {e}")
            raise

    def decode_transaction(self, tx_data: dict) -> dict:
        """
        Decode transaction into structured format.

        Step 2 of pipeline: Apply protocol-specific decoders to extract
        structured information about the transaction action.

        Args:
            tx_data: Raw transaction data from fetch_transaction()

        Returns:
            Decoded transaction dictionary with protocol, action, amounts, etc.

        Notes:
            Tries multiple decoders in sequence:
            1. ETH transfer (simple value transfer)
            2. ERC-20 transfers (token operations)
            3. Uniswap V2 swaps (AMM trading)
            4. Uniswap V3 swaps (concentrated liquidity)

            Returns first successful decode, or basic structure if none match.
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: DECODE TRANSACTION")
        logger.info("=" * 60)

        tx = tx_data["transaction"]
        receipt = tx_data["receipt"]

        logger.info("Trying protocol-specific decoders...")

        decoded_results = []

        # Try ETH transfer decoder
        logger.info("\n  [1/4] Checking ETH transfer...")
        try:
            eth_result = decode_eth_transfer(tx, receipt)
            if eth_result:
                decoded_results.append(eth_result)
                logger.info(
                    f"    ✓ ETH transfer detected: {eth_result.get('amount_eth', 0)} ETH"
                )
        except Exception as e:
            logger.debug(f"    ✗ ETH decoder error: {e}")

        # Try ERC-20 decoder
        logger.info("  [2/4] Checking ERC-20 transfers...")
        try:
            erc20_results = decode_erc20_transfers(tx, receipt, self.w3_manager.w3)
            if erc20_results:
                decoded_results.extend(erc20_results)
                logger.info(
                    f"    ✓ ERC-20 transfers detected: {len(erc20_results)} transfer(s)"
                )
                for result in erc20_results:
                    logger.info(
                        f"      - {result.get('amount_formatted', 'unknown')} "
                        f"{result.get('token_symbol', 'UNKNOWN')}"
                    )
        except Exception as e:
            logger.debug(f"    ✗ ERC-20 decoder error: {e}")

        # Try Uniswap V2 decoder
        logger.info("  [3/4] Checking Uniswap V2 swaps...")
        try:
            uniswap_v2_results = decode_uniswap_v2_swaps(
                tx, receipt, self.w3_manager.w3
            )
            if uniswap_v2_results:
                decoded_results.extend(uniswap_v2_results)
                logger.info(
                    f"    ✓ Uniswap V2 swaps detected: {len(uniswap_v2_results)} swap(s)"
                )
                for result in uniswap_v2_results:
                    logger.info(
                        f"      - {result.get('token_in_symbol', 'UNKNOWN')} → "
                        f"{result.get('token_out_symbol', 'UNKNOWN')}"
                    )
        except Exception as e:
            logger.debug(f"    ✗ Uniswap V2 decoder error: {e}")

        # Try Uniswap V3 decoder
        logger.info("  [4/4] Checking Uniswap V3 swaps...")
        try:
            uniswap_v3_results = decode_uniswap_v3_swaps(
                tx, receipt, self.w3_manager.w3
            )
            if uniswap_v3_results:
                decoded_results.extend(uniswap_v3_results)
                logger.info(
                    f"    ✓ Uniswap V3 swaps detected: {len(uniswap_v3_results)} swap(s)"
                )
                for result in uniswap_v3_results:
                    logger.info(
                        f"      - {result.get('token_in_symbol', 'UNKNOWN')} → "
                        f"{result.get('token_out_symbol', 'UNKNOWN')}"
                    )
        except Exception as e:
            logger.debug(f"    ✗ Uniswap V3 decoder error: {e}")

        # Select best decode result
        if decoded_results:
            # Prioritize: Uniswap > ERC-20 > ETH
            # (more complex protocols are usually more informative)
            decoded = decoded_results[-1]  # Take last (most complex)
            logger.info(
                f"\n✓ Transaction decoded as: {decoded.get('protocol', 'unknown')}"
            )
        else:
            logger.warning("\n⚠ No decoder matched - unknown transaction type")
            decoded = {
                "tx_hash": tx.get("hash", "unknown"),
                "protocol": "unknown",
                "action": "unknown",
                "status": "success" if receipt.get("status") == 1 else "failed",
            }

        return decoded

    def format_prompt(self, decoded_tx: dict) -> str:
        """
        Format decoded transaction as model prompt.

        Step 3 of pipeline: Convert decoded transaction into the prompt
        format used during training (Alpaca-style instruction + input).

        Args:
            decoded_tx: Decoded transaction dictionary

        Returns:
            Formatted prompt string ready for model inference
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: FORMAT PROMPT")
        logger.info("=" * 60)

        logger.info("Creating prompt in Alpaca instruction format...")

        # Format transaction as JSON
        tx_json = json.dumps(decoded_tx, indent=None, separators=(",", ":"))

        # Create prompt matching training format
        prompt = (
            f"{DEFAULT_INSTRUCTION}\n\n"
            f"Transaction data:\n{tx_json}\n\n"
            f"Extracted intent:"
        )

        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Estimated tokens: ~{len(prompt) // 4}")
        logger.info(f"\nPrompt preview:\n{'-' * 60}")
        logger.info(prompt[:300] + "...")
        logger.info("-" * 60)

        return prompt

    def run_model_inference(self, prompt: str) -> str:
        """
        Run model inference to extract intent.

        Step 4 of pipeline: Generate structured intent using the fine-tuned
        model with optimal inference settings.

        Args:
            prompt: Formatted prompt string

        Returns:
            Generated text output from model

        Notes:
            Inference settings:
            - Temperature: 0.1 (low for deterministic, focused output)
            - torch.no_grad(): Disables gradient computation (saves memory)
            - Max tokens: 512 (sufficient for JSON intent)
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: RUN MODEL INFERENCE")
        logger.info("=" * 60)

        logger.info("Generating intent with fine-tuned model...")
        logger.info("Settings:")
        logger.info("  - Temperature: 0.1 (deterministic)")
        logger.info("  - Max tokens: 512")
        logger.info("  - Sampling: Greedy (temperature < 0.1)")

        try:
            start_time = time.time()

            # Run inference with no gradient computation
            with torch.no_grad():
                generated_text = run_inference(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_new_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                )

            inference_time = time.time() - start_time

            logger.info(f"✓ Inference completed in {inference_time:.2f}s")
            logger.info(f"Generated {len(generated_text)} characters")
            logger.info(f"\nRaw output:\n{'-' * 60}")
            logger.info(
                generated_text[:500] + ("..." if len(generated_text) > 500 else "")
            )
            logger.info("-" * 60)

            return generated_text

        except Exception as e:
            logger.error(f"✗ Inference failed: {e}")
            raise

    def parse_intent(self, generated_text: str) -> dict:
        """
        Parse generated text into structured intent JSON.

        Step 5 of pipeline: Extract and validate JSON structure from
        model output, with error handling for malformed outputs.

        Args:
            generated_text: Raw text output from model

        Returns:
            Parsed intent dictionary

        Notes:
            Handles common errors:
            - Malformed JSON (missing braces, quotes)
            - Extra text before/after JSON
            - Missing required fields

            Returns error dictionary if parsing fails, preserving raw output.
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: PARSE INTENT")
        logger.info("=" * 60)

        logger.info("Parsing JSON from generated text...")

        # Use evaluator's parse function (handles edge cases)
        intent = parse_json_output(generated_text)

        if intent is None:
            logger.warning("⚠ Failed to parse valid JSON")
            logger.info("This can happen if model generates malformed output")
            intent = {
                "error": "Failed to parse JSON",
                "raw_output": generated_text,
            }
        else:
            logger.info("✓ JSON parsed successfully")
            logger.info(f"\nExtracted Intent:")
            logger.info(f"  Action: {intent.get('action', 'unknown')}")
            logger.info(f"  Protocol: {intent.get('protocol', 'unknown')}")
            logger.info(f"  Assets: {intent.get('assets', [])}")
            logger.info(f"  Outcome: {intent.get('outcome', 'unknown')}")

        return intent

    def analyze(self, tx_hash: str) -> dict:
        """
        Run complete analysis pipeline.

        Main entry point that orchestrates all steps:
        1. Fetch transaction from RPC
        2. Decode using protocol decoders
        3. Format as model prompt
        4. Run inference
        5. Parse intent

        Args:
            tx_hash: Transaction hash to analyze

        Returns:
            Complete analysis result with all intermediate steps

        Raises:
            Exception: If any pipeline step fails
        """
        # Ensure initialized
        if self.model is None or self.w3_manager is None:
            self.initialize()

        # Run pipeline
        tx_data = self.fetch_transaction(tx_hash)
        decoded_tx = self.decode_transaction(tx_data)
        prompt = self.format_prompt(decoded_tx)
        generated_text = self.run_model_inference(prompt)
        intent = self.parse_intent(generated_text)

        # Compile results
        result = {
            "transaction_hash": tx_hash,
            "decoded_transaction": decoded_tx,
            "model_prompt": prompt,
            "model_output_raw": generated_text,
            "extracted_intent": intent,
        }

        return result


def save_result(result: dict, output_path: Path) -> None:
    """Save analysis result to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n✓ Full result saved to: {output_path}")


def print_summary(result: dict) -> None:
    """Print human-readable summary of analysis."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    decoded = result["decoded_transaction"]
    intent = result["extracted_intent"]

    print(f"\nTransaction: {result['transaction_hash']}")
    print(f"\nDecoded Information:")
    print(f"  Protocol: {decoded.get('protocol', 'unknown')}")
    print(f"  Action: {decoded.get('action', 'unknown')}")
    print(f"  Status: {decoded.get('status', 'unknown')}")

    print(f"\nExtracted Intent (from fine-tuned model):")
    print(json.dumps(intent, indent=2))

    print("\n" + "=" * 60)


@click.command()
@click.option(
    "--tx-hash",
    type=str,
    required=True,
    help="Transaction hash to analyze",
)
@click.option(
    "--rpc-url",
    type=str,
    required=True,
    help="Ethereum RPC endpoint URL",
)
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to fine-tuned model directory",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file path (optional)",
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
    tx_hash: str,
    rpc_url: str,
    model: Path,
    output: Path | None,
    log_level: str,
) -> None:
    """
    Analyze Ethereum transaction with complete pipeline demonstration.

    This script runs the full workflow from raw transaction hash to
    structured intent extraction, with detailed logging at each step.

    Example:
        python analyze_transaction.py \\
            --tx-hash 0xabcdef1234567890... \\
            --rpc-url https://mainnet.infura.io/v3/YOUR_KEY \\
            --model models/fine-tuned/eth-intent-extractor-v1 \\
            --output outputs/predictions/analysis.json

    Educational Notes:
        Each pipeline step is logged separately to show:
        - What data is fetched/processed
        - Which decoders are tried
        - How prompts are formatted
        - Model inference settings
        - Intent parsing logic

        This makes the script ideal for learning about fine-tuning
        workflows and blockchain data processing.
    """
    # Setup logging
    setup_logging(log_level)

    logger.info("\n" + "=" * 80)
    logger.info(" ETHEREUM TRANSACTION ANALYSIS PIPELINE ".center(80, "="))
    logger.info("=" * 80)
    logger.info(f"Transaction: {tx_hash}")
    logger.info(f"RPC Endpoint: {rpc_url}")
    logger.info(f"Model: {model}")
    logger.info("=" * 80 + "\n")

    try:
        # Create analyzer
        analyzer = TransactionAnalyzer(rpc_url=rpc_url, model_path=model)

        # Run analysis
        result = analyzer.analyze(tx_hash)

        # Save result if output path provided
        if output:
            save_result(result, output)

        # Print summary
        print_summary(result)

        logger.info("\n✓ Analysis complete!")

    except KeyboardInterrupt:
        logger.info("\n⚠ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n✗ Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
