"""
Minimal unit tests for core extraction functionality.

Tests only critical paths:
- Web3 connection validation
- Retry logic with exponential backoff
- ABI loading
- Transaction hash parsing
- Address checksumming
"""

import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
from web3 import Web3
from web3.exceptions import Web3Exception

from eth_finetuning.extraction.core.utils import Web3ConnectionManager, load_abi
from eth_finetuning.extraction.core.fetcher import load_transaction_hashes


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_tx_hashes_file(fixtures_dir: Path) -> Path:
    """Return path to sample transaction hashes file."""
    return fixtures_dir / "sample_tx_hashes.txt"


@pytest.fixture
def mock_web3():
    """Create a mock Web3 instance for testing."""
    mock = MagicMock()
    mock.is_connected.return_value = True
    return mock


# Core Connection Tests
class TestWeb3Connection:
    """Test core Web3 connection functionality."""

    def test_placeholder_url_rejected(self):
        """Test that placeholder RPC URL is rejected."""
        with pytest.raises(ValueError, match="Invalid RPC URL"):
            Web3ConnectionManager(rpc_url="PLACEHOLDER_RPC_URL")

    def test_retry_logic_succeeds_after_failures(self, mock_web3):
        """Test that retry logic works after transient failures."""
        with patch("eth_finetuning.extraction.core.utils.Web3") as mock_web3_class:
            mock_web3_class.return_value = mock_web3
            manager = Web3ConnectionManager(
                rpc_url="http://localhost:8545",
                max_retries=3,
                backoff_factor=0.01,  # Fast backoff for testing
            )

            # Mock function that fails twice then succeeds
            mock_func = Mock(
                side_effect=[
                    Web3Exception("Error 1"),
                    Web3Exception("Error 2"),
                    "success",
                ]
            )

            result = manager.retry_with_backoff(mock_func)
            assert result == "success"
            assert mock_func.call_count == 3

    def test_retry_logic_fails_after_max_retries(self, mock_web3):
        """Test that retry logic raises exception after max retries."""
        with patch("eth_finetuning.extraction.core.utils.Web3") as mock_web3_class:
            mock_web3_class.return_value = mock_web3
            manager = Web3ConnectionManager(
                rpc_url="http://localhost:8545",
                max_retries=3,
                backoff_factor=0.01,
            )

            mock_func = Mock(side_effect=Web3Exception("Persistent error"))

            with pytest.raises(Web3Exception, match="Persistent error"):
                manager.retry_with_backoff(mock_func)

            assert mock_func.call_count == 3


# ABI Loading Tests
class TestABILoading:
    """Test ABI loading functionality."""

    def test_load_abi_success(self, tmp_path):
        """Test successful ABI loading."""
        abi_file = tmp_path / "test.json"
        abi_data = [{"type": "function", "name": "transfer"}]

        with open(abi_file, "w") as f:
            json.dump(abi_data, f)

        result = load_abi("test", abis_dir=tmp_path)
        assert result == abi_data

    def test_load_abi_file_not_found(self, tmp_path):
        """Test that missing ABI raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ABI file not found"):
            load_abi("nonexistent", abis_dir=tmp_path)


# Transaction Hash Parsing Tests
class TestTransactionHashParsing:
    """Test transaction hash file parsing."""

    def test_load_valid_hashes_from_file(self, sample_tx_hashes_file):
        """Test loading valid transaction hashes."""
        hashes = load_transaction_hashes(sample_tx_hashes_file)

        assert len(hashes) == 3
        assert all(h.startswith("0x") for h in hashes)
        assert all(len(h) == 66 for h in hashes)

    def test_load_hashes_skips_invalid_lines(self, tmp_path):
        """Test that invalid hashes and comments are skipped."""
        hash_file = tmp_path / "hashes.txt"

        with open(hash_file, "w") as f:
            f.write("# Comment line\n")
            f.write("\n")  # Empty line
            f.write("0x1234\n")  # Too short - invalid
            f.write(
                "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef\n"
            )  # Valid

        hashes = load_transaction_hashes(hash_file)
        assert len(hashes) == 1  # Only one valid hash


# Address Checksumming Tests
class TestAddressChecksumming:
    """Test that addresses are properly checksummed."""

    def test_serialize_transaction_checksums_addresses(self):
        """Test transaction serialization checksums addresses."""
        # Test the static method directly without mocking Web3
        mock_tx = {
            "hash": b"0x1234",
            "blockNumber": 100,
            "from": "0x742d35cc6634c0532925a3b844bc9e7595f0beb1",  # lowercase
            "to": "0x8f3cf7ad23cd3cadbD9735aff958023239c6a063",  # mixed case
            "value": 1000,
            "input": b"0x",
            "gas": 21000,
            "gasPrice": 30000,
            "blockHash": b"0xabcd",
            "transactionIndex": 0,
            "nonce": 1,
            "type": 2,
            "chainId": 1,
        }

        result = Web3ConnectionManager._serialize_transaction(mock_tx)

        # Verify addresses are valid checksummed format
        assert Web3.is_checksum_address(result["from"])
        assert Web3.is_checksum_address(result["to"])
        # Verify they match the original addresses (case-insensitive)
        assert result["from"].lower() == "0x742d35cc6634c0532925a3b844bc9e7595f0beb1"
        assert (
            result["to"].lower() == "0x8f3cf7ad23cd3cadbD9735aff958023239c6a063".lower()
        )
