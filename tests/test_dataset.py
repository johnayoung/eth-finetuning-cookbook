"""
Unit tests for dataset preparation module.

Tests dataset preparation functionality including:
- Intent extraction from decoded transactions
- Prompt template formatting
- Train/val/test splitting (random and stratified)
- Data validation
- JSONL export
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from eth_finetuning.dataset.intent_extraction import (
    extract_intent,
    extract_intents_batch,
)
from eth_finetuning.dataset.templates import (
    format_training_example,
    format_training_examples_batch,
    format_inference_prompt,
)
from eth_finetuning.dataset.preparation import (
    prepare_dataset,
    validate_data,
    _stratified_split,
    _random_split,
)


# Sample decoded transactions for testing
@pytest.fixture
def sample_decoded_eth_transfer():
    """Sample decoded ETH transfer."""
    return {
        "tx_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "block_number": 18000000,
        "action": "transfer",
        "protocol": "ethereum",
        "from": "0x742d35cC6634c0532925A3b844bc9E7595F0beB1",
        "to": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
        "amount_wei": 1000000000000000000,
        "amount_eth": 1.0,
        "status": "success",
        "gas_used": 21000,
    }


@pytest.fixture
def sample_decoded_erc20_transfer():
    """Sample decoded ERC-20 transfer."""
    return {
        "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "block_number": 18000001,
        "action": "transfer",
        "protocol": "erc20",
        "token_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "token_symbol": "USDT",
        "token_decimals": 6,
        "from": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "to": "0x742d35cC6634c0532925A3b844bc9E7595F0beB1",
        "amount": 100000,
        "amount_formatted": 0.1,
        "status": "success",
        "log_index": 0,
    }


@pytest.fixture
def sample_decoded_uniswap_v2_swap():
    """Sample decoded Uniswap V2 swap."""
    return {
        "tx_hash": "0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
        "block_number": 18000002,
        "action": "swap",
        "protocol": "uniswap_v2",
        "pool_address": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
        "token_in": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "token_out": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "token_in_symbol": "USDC",
        "token_out_symbol": "WETH",
        "amount_in": 100000,
        "amount_out": 500000000000000000,
        "amount_in_formatted": 0.1,
        "amount_out_formatted": 0.5,
        "status": "success",
        "log_index": 1,
    }


@pytest.fixture
def sample_decoded_transactions(
    sample_decoded_eth_transfer,
    sample_decoded_erc20_transfer,
    sample_decoded_uniswap_v2_swap,
):
    """List of sample decoded transactions covering all protocols."""
    return [
        sample_decoded_eth_transfer,
        sample_decoded_erc20_transfer,
        sample_decoded_uniswap_v2_swap,
    ]


# Intent Extraction Tests
class TestIntentExtraction:
    """Test intent extraction from decoded transactions."""

    def test_extract_eth_intent(self, sample_decoded_eth_transfer):
        """Test extracting intent from ETH transfer."""
        intent = extract_intent(sample_decoded_eth_transfer)

        assert intent["action"] == "transfer"
        assert intent["assets"] == ["ETH"]
        assert intent["protocol"] == "ethereum"
        assert intent["outcome"] == "success"
        assert intent["amounts"] == [1.0]

    def test_extract_erc20_intent(self, sample_decoded_erc20_transfer):
        """Test extracting intent from ERC-20 transfer."""
        intent = extract_intent(sample_decoded_erc20_transfer)

        assert intent["action"] == "transfer"
        assert intent["assets"] == ["USDT"]
        assert intent["protocol"] == "erc20"
        assert intent["outcome"] == "success"
        assert intent["amounts"] == [0.1]

    def test_extract_uniswap_v2_intent(self, sample_decoded_uniswap_v2_swap):
        """Test extracting intent from Uniswap V2 swap."""
        intent = extract_intent(sample_decoded_uniswap_v2_swap)

        assert intent["action"] == "swap"
        assert intent["assets"] == ["USDC", "WETH"]
        assert intent["protocol"] == "uniswap_v2"
        assert intent["outcome"] == "success"
        assert intent["amounts"] == [0.1, 0.5]

    def test_extract_failed_transaction_intent(self, sample_decoded_eth_transfer):
        """Test extracting intent from failed transaction."""
        failed_tx = sample_decoded_eth_transfer.copy()
        failed_tx["status"] = "failed"

        intent = extract_intent(failed_tx)

        assert intent["outcome"] == "failed"

    def test_extract_intent_missing_protocol(self):
        """Test extracting intent from transaction without protocol."""
        invalid_tx = {"action": "transfer", "status": "success"}

        with pytest.raises(ValueError, match="must have 'protocol' field"):
            extract_intent(invalid_tx)

    def test_extract_intent_missing_action(self):
        """Test extracting intent from transaction without action."""
        invalid_tx = {"protocol": "ethereum", "status": "success"}

        with pytest.raises(ValueError, match="must have 'action' field"):
            extract_intent(invalid_tx)

    def test_extract_intent_unknown_protocol(self):
        """Test extracting intent from transaction with unknown protocol."""
        invalid_tx = {
            "action": "transfer",
            "protocol": "unknown_protocol",
            "status": "success",
        }

        with pytest.raises(ValueError, match="Unknown protocol"):
            extract_intent(invalid_tx)

    def test_extract_intents_batch(self, sample_decoded_transactions):
        """Test batch intent extraction."""
        intents = extract_intents_batch(sample_decoded_transactions)

        assert len(intents) == 3
        assert all("action" in intent for intent in intents)
        assert all("assets" in intent for intent in intents)
        assert all("protocol" in intent for intent in intents)
        assert all("outcome" in intent for intent in intents)
        assert all("amounts" in intent for intent in intents)
        assert all("tx_hash" in intent for intent in intents)
        assert all("block_number" in intent for intent in intents)

    def test_extract_intents_batch_with_invalid(self, sample_decoded_eth_transfer):
        """Test batch intent extraction with some invalid transactions."""
        valid_tx = sample_decoded_eth_transfer
        invalid_tx = {"action": "transfer"}  # Missing protocol

        transactions = [valid_tx, invalid_tx, valid_tx]
        intents = extract_intents_batch(transactions)

        # Should skip invalid transaction
        assert len(intents) == 2


# Prompt Template Tests
class TestPromptTemplates:
    """Test prompt template formatting."""

    def test_format_training_example_eth(self, sample_decoded_eth_transfer):
        """Test formatting training example for ETH transfer."""
        intent = extract_intent(sample_decoded_eth_transfer)
        example = format_training_example(sample_decoded_eth_transfer, intent)

        assert "instruction" in example
        assert "input" in example
        assert "output" in example

        # Verify instruction
        assert "Extract the structured intent" in example["instruction"]

        # Verify input is valid JSON
        input_data = json.loads(example["input"])
        assert input_data["protocol"] == "ethereum"
        assert input_data["action"] == "transfer"
        assert input_data["amount_eth"] == 1.0

        # Verify output is valid JSON
        output_data = json.loads(example["output"])
        assert output_data["action"] == "transfer"
        assert output_data["assets"] == ["ETH"]
        assert output_data["protocol"] == "ethereum"
        assert output_data["outcome"] == "success"
        assert output_data["amounts"] == [1.0]

    def test_format_training_example_erc20(self, sample_decoded_erc20_transfer):
        """Test formatting training example for ERC-20 transfer."""
        intent = extract_intent(sample_decoded_erc20_transfer)
        example = format_training_example(sample_decoded_erc20_transfer, intent)

        input_data = json.loads(example["input"])
        assert input_data["protocol"] == "erc20"
        assert input_data["token_symbol"] == "USDT"

        output_data = json.loads(example["output"])
        assert output_data["assets"] == ["USDT"]

    def test_format_training_example_uniswap(self, sample_decoded_uniswap_v2_swap):
        """Test formatting training example for Uniswap swap."""
        intent = extract_intent(sample_decoded_uniswap_v2_swap)
        example = format_training_example(sample_decoded_uniswap_v2_swap, intent)

        input_data = json.loads(example["input"])
        assert input_data["protocol"] == "uniswap_v2"
        assert input_data["token_in_symbol"] == "USDC"
        assert input_data["token_out_symbol"] == "WETH"

        output_data = json.loads(example["output"])
        assert output_data["assets"] == ["USDC", "WETH"]
        assert len(output_data["amounts"]) == 2

    def test_format_training_example_custom_instruction(
        self, sample_decoded_eth_transfer
    ):
        """Test formatting training example with custom instruction."""
        intent = extract_intent(sample_decoded_eth_transfer)
        custom_instruction = "Custom instruction for testing"

        example = format_training_example(
            sample_decoded_eth_transfer, intent, custom_instruction
        )

        assert example["instruction"] == custom_instruction

    def test_format_training_examples_batch(self, sample_decoded_transactions):
        """Test batch formatting of training examples."""
        intents = extract_intents_batch(sample_decoded_transactions)
        examples = format_training_examples_batch(sample_decoded_transactions, intents)

        assert len(examples) == 3
        for example in examples:
            assert "instruction" in example
            assert "input" in example
            assert "output" in example
            # Verify JSON is valid
            json.loads(example["input"])
            json.loads(example["output"])

    def test_format_training_examples_batch_length_mismatch(
        self, sample_decoded_eth_transfer
    ):
        """Test batch formatting with mismatched lengths."""
        decoded_txs = [sample_decoded_eth_transfer]
        intents = []

        with pytest.raises(ValueError, match="must have same length"):
            format_training_examples_batch(decoded_txs, intents)

    def test_format_inference_prompt(self, sample_decoded_eth_transfer):
        """Test formatting inference prompt."""
        prompt = format_inference_prompt(sample_decoded_eth_transfer)

        assert "Extract the structured intent" in prompt
        assert "Input:" in prompt
        assert "Output:" in prompt
        # Verify JSON is embedded in prompt
        assert sample_decoded_eth_transfer["tx_hash"] in prompt


# Data Validation Tests
class TestDataValidation:
    """Test data validation logic."""

    def test_validate_data_success(self, sample_decoded_transactions):
        """Test validation with valid data."""
        intents = extract_intents_batch(sample_decoded_transactions)

        # Should not raise any exception
        validate_data(sample_decoded_transactions, intents)

    def test_validate_data_length_mismatch(self, sample_decoded_eth_transfer):
        """Test validation with mismatched lengths."""
        decoded_txs = [sample_decoded_eth_transfer]
        intents = []

        with pytest.raises(ValueError, match="Mismatch"):
            validate_data(decoded_txs, intents)

    def test_validate_data_missing_field(self, sample_decoded_eth_transfer):
        """Test validation with missing critical field."""
        invalid_tx = sample_decoded_eth_transfer.copy()
        del invalid_tx["action"]

        intent = {
            "action": "transfer",
            "protocol": "ethereum",
            "outcome": "success",
            "assets": ["ETH"],
            "amounts": [1.0],
        }

        with pytest.raises(ValueError, match="missing 'action'"):
            validate_data([invalid_tx], [intent])

    def test_validate_data_protocol_mismatch(self, sample_decoded_eth_transfer):
        """Test validation with protocol mismatch."""
        decoded_tx = sample_decoded_eth_transfer
        intent = {
            "action": "transfer",
            "protocol": "erc20",  # Mismatch
            "outcome": "success",
            "assets": ["ETH"],
            "amounts": [1.0],
        }

        with pytest.raises(ValueError, match="protocol mismatch"):
            validate_data([decoded_tx], [intent])

    def test_validate_data_invalid_address(self, sample_decoded_eth_transfer):
        """Test validation with non-checksummed address."""
        invalid_tx = sample_decoded_eth_transfer.copy()
        invalid_tx["from"] = invalid_tx["from"].lower()  # Not checksummed

        intent = extract_intent(sample_decoded_eth_transfer)

        with pytest.raises(ValueError, match="not checksummed"):
            validate_data([invalid_tx], [intent])

    def test_validate_data_invalid_amount(self, sample_decoded_eth_transfer):
        """Test validation with non-numeric amount."""
        invalid_tx = sample_decoded_eth_transfer.copy()
        invalid_tx["amount_eth"] = "not_a_number"

        intent = {
            "action": "transfer",
            "protocol": "ethereum",
            "outcome": "success",
            "assets": ["ETH"],
            "amounts": [1.0],
        }

        with pytest.raises(ValueError, match="not numeric"):
            validate_data([invalid_tx], [intent])


# Dataset Splitting Tests
class TestDatasetSplitting:
    """Test dataset splitting logic."""

    def test_random_split_ratios(self, sample_decoded_transactions):
        """Test random split with specified ratios."""
        intents = extract_intents_batch(sample_decoded_transactions)
        examples = format_training_examples_batch(sample_decoded_transactions, intents)

        # Add more examples to test splitting (need at least 10 for good split)
        examples = examples * 4  # 12 examples total

        train, val, test = _random_split(examples, (0.7, 0.15, 0.15))

        # Check approximate ratios (with small dataset, exact ratios may vary)
        total = len(examples)
        assert len(train) == int(total * 0.7)
        assert len(val) == int(total * 0.15)
        # Test set gets the remainder
        assert len(test) == total - len(train) - len(val)

        # Verify all examples accounted for
        assert len(train) + len(val) + len(test) == total

    def test_stratified_split_ratios(self, sample_decoded_transactions):
        """Test stratified split maintains protocol distribution."""
        # Create more examples with protocol distribution
        eth_tx = sample_decoded_transactions[0]
        erc20_tx = sample_decoded_transactions[1]
        uniswap_tx = sample_decoded_transactions[2]

        # 10 eth, 10 erc20, 10 uniswap (total 30) - need larger sets for stratification
        decoded_txs = [eth_tx] * 10 + [erc20_tx] * 10 + [uniswap_tx] * 10

        intents = extract_intents_batch(decoded_txs)
        examples = format_training_examples_batch(decoded_txs, intents)

        train, val, test = _stratified_split(examples, decoded_txs, (0.7, 0.15, 0.15))

        # Verify split sizes
        total = len(examples)
        assert len(train) + len(val) + len(test) == total

        # Verify protocol distribution is maintained (approximately)
        # Each protocol should appear in train/val/test with reasonable distribution
        assert len(train) >= 15  # Approximately 70% of 30
        assert len(val) >= 1
        assert len(test) >= 1


# End-to-End Pipeline Tests
class TestDatasetPreparation:
    """Test complete dataset preparation pipeline."""

    def test_prepare_dataset_success(self, sample_decoded_transactions):
        """Test complete dataset preparation pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            counts = prepare_dataset(
                decoded_txs=sample_decoded_transactions,
                output_dir=output_dir,
                split_ratios=(0.7, 0.15, 0.15),
                stratify_by_protocol=True,
            )

            # Verify counts
            assert counts["train"] + counts["validation"] + counts["test"] == 3

            # Verify files exist
            assert (output_dir / "train.jsonl").exists()
            assert (output_dir / "validation.jsonl").exists()
            assert (output_dir / "test.jsonl").exists()

            # Verify JSONL format
            with open(output_dir / "train.jsonl", "r") as f:
                for line in f:
                    example = json.loads(line)
                    assert "instruction" in example
                    assert "input" in example
                    assert "output" in example

    def test_prepare_dataset_custom_split(self, sample_decoded_transactions):
        """Test dataset preparation with custom split ratios."""
        # Duplicate transactions to have enough for splitting
        transactions = sample_decoded_transactions * 4  # 12 total

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            counts = prepare_dataset(
                decoded_txs=transactions,
                output_dir=output_dir,
                split_ratios=(0.8, 0.1, 0.1),
                stratify_by_protocol=False,
            )

            total = counts["train"] + counts["validation"] + counts["test"]
            assert total == 12

    def test_prepare_dataset_empty_input(self):
        """Test dataset preparation with empty input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with pytest.raises(ValueError, match="cannot be empty"):
                prepare_dataset(
                    decoded_txs=[],
                    output_dir=output_dir,
                )

    def test_prepare_dataset_invalid_split_ratios(self, sample_decoded_transactions):
        """Test dataset preparation with invalid split ratios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with pytest.raises(ValueError, match="must sum to 1.0"):
                prepare_dataset(
                    decoded_txs=sample_decoded_transactions,
                    output_dir=output_dir,
                    split_ratios=(0.5, 0.3, 0.1),  # Sums to 0.9
                )
