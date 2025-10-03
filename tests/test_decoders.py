"""
Unit tests for transaction decoders (ETH and ERC-20).

Tests decoder functionality using fixture data from tests/fixtures/sample_transactions.json.
Covers:
- ETH transfer decoding
- ERC-20 Transfer event decoding
- Edge cases (failed transactions, zero-value transfers, missing metadata)
- Address checksumming validation
"""

import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
from web3 import Web3

from eth_finetuning.extraction.decoders.eth import (
    decode_eth_transfer,
    decode_eth_transfers_batch,
)
from eth_finetuning.extraction.decoders.erc20 import (
    decode_erc20_transfers,
    decode_erc20_transfers_batch,
    _get_token_metadata,
)


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_transactions(fixtures_dir: Path) -> list[dict]:
    """Load sample transactions from fixture file."""
    fixture_path = fixtures_dir / "sample_transactions.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


@pytest.fixture
def mock_web3():
    """Create a mock Web3 instance for testing."""
    mock = MagicMock()
    mock.is_connected.return_value = True

    # Mock contract interface
    mock_contract = MagicMock()
    mock.eth.contract.return_value = mock_contract

    return mock


# ETH Transfer Decoder Tests
class TestETHTransferDecoder:
    """Test ETH transfer decoding functionality."""

    def test_decode_simple_eth_transfer(self, sample_transactions):
        """Test decoding a successful simple ETH transfer."""
        # First transaction in fixtures is a simple ETH transfer
        tx = sample_transactions[0]
        receipt = {
            "status": tx["status"],
            "gasUsed": tx["gas_used"],
            "logs": tx["logs"],
        }

        decoded = decode_eth_transfer(tx, receipt)

        assert decoded is not None
        assert decoded["action"] == "transfer"
        assert decoded["protocol"] == "ethereum"
        assert decoded["from"] == Web3.to_checksum_address(tx["from"])
        assert decoded["to"] == Web3.to_checksum_address(tx["to"])
        assert decoded["amount_wei"] == tx["value"]
        assert decoded["amount_eth"] == 1.0  # 1 ETH in the fixture
        assert decoded["status"] == "success"
        assert decoded["gas_used"] == tx["gas_used"]

    def test_decode_failed_eth_transfer(self, sample_transactions):
        """Test decoding a failed ETH transfer."""
        tx = sample_transactions[0].copy()
        receipt = {
            "status": 0,  # Failed transaction
            "gasUsed": tx["gas_used"],
            "logs": [],
        }

        decoded = decode_eth_transfer(tx, receipt)

        assert decoded is not None
        assert decoded["status"] == "failed"
        assert decoded["amount_wei"] == tx["value"]

    def test_decode_zero_value_transfer(self, sample_transactions):
        """Test decoding ETH transfer with zero value."""
        tx = sample_transactions[0].copy()
        tx["value"] = 0
        receipt = {
            "status": 1,
            "gasUsed": tx["gas_used"],
            "logs": [],
        }

        decoded = decode_eth_transfer(tx, receipt)

        assert decoded is not None
        assert decoded["amount_wei"] == 0
        assert decoded["amount_eth"] == 0.0

    def test_decode_contract_call_returns_none(self, sample_transactions):
        """Test that transactions with input data are not decoded as simple transfers."""
        # Second transaction has input data (ERC-20 transfer)
        tx = sample_transactions[1]
        receipt = {
            "status": tx["status"],
            "gasUsed": tx["gas_used"],
            "logs": tx["logs"],
        }

        decoded = decode_eth_transfer(tx, receipt)

        # Should return None because it has input data
        assert decoded is None

    def test_decode_contract_creation_returns_none(self, sample_transactions):
        """Test that contract creation transactions are not decoded."""
        tx = sample_transactions[0].copy()
        tx["to"] = None  # Contract creation has no recipient
        receipt = {
            "status": 1,
            "gasUsed": 100000,
            "logs": [],
        }

        decoded = decode_eth_transfer(tx, receipt)

        assert decoded is None

    def test_addresses_are_checksummed(self, sample_transactions):
        """Test that all addresses are properly checksummed."""
        tx = sample_transactions[0].copy()
        # Use lowercase addresses to test checksumming
        tx["from"] = tx["from"].lower()
        tx["to"] = tx["to"].lower()

        receipt = {
            "status": 1,
            "gasUsed": tx["gas_used"],
            "logs": [],
        }

        decoded = decode_eth_transfer(tx, receipt)

        assert decoded is not None
        # Verify addresses are checksummed (contain both upper and lowercase)
        assert decoded["from"] != decoded["from"].lower()
        assert decoded["to"] != decoded["to"].lower()
        # Verify they're valid checksum addresses
        assert Web3.is_checksum_address(decoded["from"])
        assert Web3.is_checksum_address(decoded["to"])

    def test_batch_decode_eth_transfers(self, sample_transactions):
        """Test batch decoding of multiple transactions."""
        # Use first transaction (ETH transfer) twice
        tx = sample_transactions[0]
        receipt = {
            "status": tx["status"],
            "gasUsed": tx["gas_used"],
            "logs": tx["logs"],
        }

        transactions_with_receipts = [(tx, receipt), (tx, receipt)]

        decoded_list = decode_eth_transfers_batch(transactions_with_receipts)

        assert len(decoded_list) == 2
        assert all(d["action"] == "transfer" for d in decoded_list)
        assert all(d["protocol"] == "ethereum" for d in decoded_list)
        assert all("tx_hash" in d for d in decoded_list)
        assert all("block_number" in d for d in decoded_list)


# ERC-20 Transfer Decoder Tests
class TestERC20TransferDecoder:
    """Test ERC-20 transfer decoding functionality."""

    def test_decode_erc20_transfer(self, sample_transactions, mock_web3):
        """Test decoding a successful ERC-20 transfer."""
        # Second transaction in fixtures is an ERC-20 transfer
        tx = sample_transactions[1]
        receipt = {
            "status": tx["status"],
            "gasUsed": tx["gas_used"],
            "logs": tx["logs"],
        }

        # Mock token metadata calls
        mock_contract = MagicMock()
        mock_contract.functions.symbol().call.return_value = "USDT"
        mock_contract.functions.decimals().call.return_value = 6
        mock_web3.eth.contract.return_value = mock_contract

        decoded_list = decode_erc20_transfers(tx, receipt, mock_web3)

        assert len(decoded_list) == 1
        decoded = decoded_list[0]

        assert decoded["action"] == "transfer"
        assert decoded["protocol"] == "erc20"
        assert decoded["token_address"] == Web3.to_checksum_address(
            "0xdAC17F958D2ee523a2206206994597C13D831ec7"
        )
        assert decoded["token_symbol"] == "USDT"
        assert decoded["token_decimals"] == 6
        assert decoded["amount"] == 100000  # Raw amount from fixture
        assert decoded["amount_formatted"] == 0.1  # 100000 / 10^6
        assert decoded["status"] == "success"

    def test_decode_erc20_failed_transaction(self, sample_transactions, mock_web3):
        """Test decoding ERC-20 transfer in failed transaction."""
        tx = sample_transactions[1]
        receipt = {
            "status": 0,  # Failed transaction
            "gasUsed": tx["gas_used"],
            "logs": tx["logs"],
        }

        # Mock token metadata
        mock_contract = MagicMock()
        mock_contract.functions.symbol().call.return_value = "USDT"
        mock_contract.functions.decimals().call.return_value = 6
        mock_web3.eth.contract.return_value = mock_contract

        decoded_list = decode_erc20_transfers(tx, receipt, mock_web3)

        assert len(decoded_list) == 1
        assert decoded_list[0]["status"] == "failed"

    def test_decode_erc20_without_metadata(self, sample_transactions, mock_web3):
        """Test decoding ERC-20 transfer when token metadata is unavailable."""
        tx = sample_transactions[1]
        receipt = {
            "status": tx["status"],
            "gasUsed": tx["gas_used"],
            "logs": tx["logs"],
        }

        # Mock metadata calls to raise exceptions (non-standard token)
        mock_contract = MagicMock()
        mock_contract.functions.symbol().call.side_effect = Exception("No symbol")
        mock_contract.functions.decimals().call.side_effect = Exception("No decimals")
        mock_web3.eth.contract.return_value = mock_contract

        decoded_list = decode_erc20_transfers(tx, receipt, mock_web3)

        assert len(decoded_list) == 1
        decoded = decoded_list[0]

        assert decoded["token_symbol"] == "UNKNOWN"
        assert decoded["token_decimals"] is None
        assert decoded["amount_formatted"] is None
        assert decoded["amount"] == 100000  # Raw amount still available

    def test_decode_multiple_erc20_transfers(self, sample_transactions, mock_web3):
        """Test decoding transaction with multiple ERC-20 transfers."""
        # Third transaction has multiple Transfer events (Uniswap swap)
        tx = sample_transactions[2]
        receipt = {
            "status": tx["status"],
            "gasUsed": tx["gas_used"],
            "logs": tx["logs"],
        }

        # Mock token metadata
        mock_contract = MagicMock()
        mock_contract.functions.symbol().call.side_effect = ["USDC", "WETH"]
        mock_contract.functions.decimals().call.side_effect = [6, 18]
        mock_web3.eth.contract.return_value = mock_contract

        decoded_list = decode_erc20_transfers(tx, receipt, mock_web3)

        # Transaction has 2 Transfer events (in, out) plus 1 Sync event (not Transfer)
        assert len(decoded_list) == 2
        assert all(d["action"] == "transfer" for d in decoded_list)
        assert all(d["protocol"] == "erc20" for d in decoded_list)

    def test_erc20_addresses_are_checksummed(self, sample_transactions, mock_web3):
        """Test that all ERC-20 addresses are properly checksummed."""
        tx = sample_transactions[1]
        receipt = {
            "status": tx["status"],
            "gasUsed": tx["gas_used"],
            "logs": tx["logs"],
        }

        # Mock token metadata
        mock_contract = MagicMock()
        mock_contract.functions.symbol().call.return_value = "USDT"
        mock_contract.functions.decimals().call.return_value = 6
        mock_web3.eth.contract.return_value = mock_contract

        decoded_list = decode_erc20_transfers(tx, receipt, mock_web3)

        assert len(decoded_list) == 1
        decoded = decoded_list[0]

        # Verify all addresses are checksummed
        assert Web3.is_checksum_address(decoded["token_address"])
        assert Web3.is_checksum_address(decoded["from"])
        assert Web3.is_checksum_address(decoded["to"])

    def test_no_logs_returns_empty_list(self, sample_transactions, mock_web3):
        """Test that transactions with no logs return empty list."""
        tx = sample_transactions[0]  # ETH transfer with no logs
        receipt = {
            "status": 1,
            "gasUsed": 21000,
            "logs": [],
        }

        decoded_list = decode_erc20_transfers(tx, receipt, mock_web3)

        assert decoded_list == []

    def test_batch_decode_erc20_transfers(self, sample_transactions, mock_web3):
        """Test batch decoding of ERC-20 transfers from multiple transactions."""
        tx = sample_transactions[1]
        receipt = {
            "status": tx["status"],
            "gasUsed": tx["gas_used"],
            "logs": tx["logs"],
        }

        # Mock token metadata
        mock_contract = MagicMock()
        mock_contract.functions.symbol().call.return_value = "USDT"
        mock_contract.functions.decimals().call.return_value = 6
        mock_web3.eth.contract.return_value = mock_contract

        transactions_with_receipts = [(tx, receipt), (tx, receipt)]

        decoded_list = decode_erc20_transfers_batch(
            transactions_with_receipts, mock_web3
        )

        assert len(decoded_list) == 2
        assert all(d["action"] == "transfer" for d in decoded_list)
        assert all(d["protocol"] == "erc20" for d in decoded_list)
        assert all("tx_hash" in d for d in decoded_list)
        assert all("block_number" in d for d in decoded_list)


# Token Metadata Tests
class TestTokenMetadata:
    """Test token metadata fetching functionality."""

    def test_get_token_metadata_success(self, mock_web3):
        """Test successful token metadata retrieval."""
        mock_contract = MagicMock()
        mock_contract.functions.symbol().call.return_value = "USDC"
        mock_contract.functions.decimals().call.return_value = 6
        mock_web3.eth.contract.return_value = mock_contract

        from eth_finetuning.extraction.core.utils import load_abi

        erc20_abi = load_abi("erc20")

        symbol, decimals = _get_token_metadata(
            mock_web3,
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            erc20_abi,
        )

        assert symbol == "USDC"
        assert decimals == 6

    def test_get_token_metadata_missing_symbol(self, mock_web3):
        """Test metadata retrieval when symbol() function fails."""
        mock_contract = MagicMock()
        mock_contract.functions.symbol().call.side_effect = Exception("No symbol")
        mock_contract.functions.decimals().call.return_value = 18
        mock_web3.eth.contract.return_value = mock_contract

        from eth_finetuning.extraction.core.utils import load_abi

        erc20_abi = load_abi("erc20")

        symbol, decimals = _get_token_metadata(
            mock_web3,
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            erc20_abi,
        )

        assert symbol == "UNKNOWN"
        assert decimals == 18

    def test_get_token_metadata_missing_decimals(self, mock_web3):
        """Test metadata retrieval when decimals() function fails."""
        mock_contract = MagicMock()
        mock_contract.functions.symbol().call.return_value = "TOKEN"
        mock_contract.functions.decimals().call.side_effect = Exception("No decimals")
        mock_web3.eth.contract.return_value = mock_contract

        from eth_finetuning.extraction.core.utils import load_abi

        erc20_abi = load_abi("erc20")

        symbol, decimals = _get_token_metadata(
            mock_web3,
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            erc20_abi,
        )

        assert symbol == "TOKEN"
        assert decimals is None

    def test_get_token_metadata_contract_creation_fails(self, mock_web3):
        """Test metadata retrieval when contract creation fails."""
        mock_web3.eth.contract.side_effect = Exception("Invalid contract")

        from eth_finetuning.extraction.core.utils import load_abi

        erc20_abi = load_abi("erc20")

        symbol, decimals = _get_token_metadata(
            mock_web3,
            "0xInvalidAddress",
            erc20_abi,
        )

        assert symbol == "UNKNOWN"
        assert decimals is None


# Uniswap Decoder Tests
class TestUniswapV2Decoder:
    """Test Uniswap V2 swap decoding functionality."""

    @pytest.fixture
    def uniswap_v2_swap_log(self):
        """Create a mock Uniswap V2 Swap event log."""
        return {
            "address": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",  # USDC-WETH pool
            "topics": [
                Web3.keccak(
                    text="Swap(address,uint256,uint256,uint256,uint256,address)"
                ).hex(),
                "0x0000000000000000000000007a250d5630b4cf539739df2c5dacb4c659f2488d",  # sender
                "0x000000000000000000000000742d35cc6634c0532925a3b844bc9e7595f0beb1",  # recipient
            ],
            "data": "0x"
            + "00000000000000000000000000000000000000000000000000000000000186a0"  # amount0In: 100000
            + "0000000000000000000000000000000000000000000000000000000000000000"  # amount1In: 0
            + "0000000000000000000000000000000000000000000000000000000000000000"  # amount0Out: 0
            + "00000000000000000000000000000000000000000000000006f05b59d3b20000",  # amount1Out
            "logIndex": 1,
        }

    def test_decode_uniswap_v2_swap(self, uniswap_v2_swap_log, mock_web3):
        """Test decoding a Uniswap V2 swap."""
        from eth_finetuning.extraction.decoders.uniswap.v2 import (
            decode_uniswap_v2_swaps,
        )

        tx = {
            "hash": "0xtest123",
            "from": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1",
            "to": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        }
        receipt = {
            "status": 1,
            "logs": [uniswap_v2_swap_log],
        }

        # Mock pool contract for token addresses
        mock_pool = MagicMock()
        mock_pool.functions.token0().call.return_value = (
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # USDC
        )
        mock_pool.functions.token1().call.return_value = (
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # WETH
        )

        # Mock token contracts
        mock_token = MagicMock()
        mock_token.functions.symbol().call.side_effect = ["USDC", "WETH"]
        mock_token.functions.decimals().call.side_effect = [6, 18]

        mock_web3.eth.contract.side_effect = [mock_pool, mock_token, mock_token]

        decoded_list = decode_uniswap_v2_swaps(tx, receipt, mock_web3)

        assert len(decoded_list) == 1
        decoded = decoded_list[0]

        assert decoded["action"] == "swap"
        assert decoded["protocol"] == "uniswap_v2"
        assert decoded["token_in_symbol"] == "USDC"
        assert decoded["token_out_symbol"] == "WETH"
        assert decoded["amount_in"] == 100000
        assert decoded["amount_out"] > 0
        assert decoded["status"] == "success"

    def test_decode_uniswap_v2_no_swap_events(self, mock_web3):
        """Test that non-swap logs are ignored."""
        from eth_finetuning.extraction.decoders.uniswap.v2 import (
            decode_uniswap_v2_swaps,
        )

        tx = {"hash": "0xtest123", "from": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1"}
        receipt = {
            "status": 1,
            "logs": [
                {
                    "address": "0xSomeContract",
                    "topics": ["0xDifferentEventSignature"],
                    "data": "0x",
                }
            ],
        }

        decoded_list = decode_uniswap_v2_swaps(tx, receipt, mock_web3)
        assert len(decoded_list) == 0


class TestUniswapV3Decoder:
    """Test Uniswap V3 swap decoding functionality."""

    @pytest.fixture
    def uniswap_v3_swap_log(self):
        """Create a mock Uniswap V3 Swap event log."""
        # amount0 = -100000 (negative, sent to pool)
        amount0_bytes = (2**256 - 100000).to_bytes(32, byteorder="big")
        # amount1 = 500000000000000000 (positive, received from pool)
        amount1_bytes = (500000000000000000).to_bytes(32, byteorder="big")
        sqrtPrice_bytes = (1 << 96).to_bytes(32, byteorder="big")  # sqrtPriceX96
        liquidity_bytes = (1000000).to_bytes(32, byteorder="big")
        tick_bytes = (100).to_bytes(32, byteorder="big")

        return {
            "address": "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",  # V3 pool
            "topics": [
                Web3.keccak(
                    text="Swap(address,address,int256,int256,uint160,uint128,int24)"
                ).hex(),
                "0x0000000000000000000000007a250d5630b4cf539739df2c5dacb4c659f2488d",  # sender
                "0x000000000000000000000000742d35cc6634c0532925a3b844bc9e7595f0beb1",  # recipient
            ],
            "data": "0x"
            + amount0_bytes.hex()
            + amount1_bytes.hex()
            + sqrtPrice_bytes.hex()
            + liquidity_bytes.hex()
            + tick_bytes.hex(),
            "logIndex": 1,
        }

    def test_decode_uniswap_v3_swap(self, uniswap_v3_swap_log, mock_web3):
        """Test decoding a Uniswap V3 swap."""
        from eth_finetuning.extraction.decoders.uniswap.v3 import (
            decode_uniswap_v3_swaps,
        )

        tx = {
            "hash": "0xtest123",
            "from": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1",
            "to": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        }
        receipt = {
            "status": 1,
            "logs": [uniswap_v3_swap_log],
        }

        # Mock pool contract
        mock_pool = MagicMock()
        mock_pool.functions.token0().call.return_value = (
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # USDC
        )
        mock_pool.functions.token1().call.return_value = (
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # WETH
        )

        # Mock token contracts
        mock_token = MagicMock()
        mock_token.functions.symbol().call.side_effect = ["USDC", "WETH"]
        mock_token.functions.decimals().call.side_effect = [6, 18]

        mock_web3.eth.contract.side_effect = [mock_pool, mock_token, mock_token]

        decoded_list = decode_uniswap_v3_swaps(tx, receipt, mock_web3)

        assert len(decoded_list) == 1
        decoded = decoded_list[0]

        assert decoded["action"] == "swap"
        assert decoded["protocol"] == "uniswap_v3"
        assert decoded["token_in_symbol"] == "USDC"
        assert decoded["token_out_symbol"] == "WETH"
        assert decoded["amount_in"] == 100000
        assert decoded["amount_out"] == 500000000000000000
        assert "sqrt_price_x96" in decoded
        assert "liquidity" in decoded
        assert "tick" in decoded
        assert decoded["status"] == "success"

    def test_decode_uniswap_v3_no_swap_events(self, mock_web3):
        """Test that non-swap logs are ignored."""
        from eth_finetuning.extraction.decoders.uniswap.v3 import (
            decode_uniswap_v3_swaps,
        )

        tx = {"hash": "0xtest123", "from": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1"}
        receipt = {"status": 1, "logs": []}

        decoded_list = decode_uniswap_v3_swaps(tx, receipt, mock_web3)
        assert len(decoded_list) == 0
