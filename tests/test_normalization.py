"""
Tests for data normalization utilities.

Tests hex field parsing and Web3 data normalization to ensure consistent
behavior across different input formats.
"""

import pytest
from hexbytes import HexBytes

from eth_finetuning.extraction.core.normalization import (
    normalize_hex_field,
    normalize_hex_string,
    normalize_log,
    normalize_web3_transaction,
    parse_log_data,
    parse_log_topics,
)


class TestNormalizeHexField:
    """Test normalize_hex_field function with various input formats."""

    def test_hex_string_with_prefix(self):
        """Test parsing hex string with 0x prefix."""
        result = normalize_hex_field("0x1234abcd")
        assert result == b"\x12\x34\xab\xcd"

    def test_hex_string_without_prefix(self):
        """Test parsing hex string without 0x prefix."""
        result = normalize_hex_field("1234abcd")
        assert result == b"\x12\x34\xab\xcd"

    def test_hexbytes_object(self):
        """Test parsing HexBytes object from Web3."""
        hex_obj = HexBytes("0x1234abcd")
        result = normalize_hex_field(hex_obj)
        assert result == b"\x12\x34\xab\xcd"

    def test_raw_bytes(self):
        """Test parsing raw bytes object."""
        raw = b"\x12\x34\xab\xcd"
        result = normalize_hex_field(raw)
        assert result == raw

    def test_empty_string(self):
        """Test parsing empty string."""
        assert normalize_hex_field("") == b""
        assert normalize_hex_field("0x") == b""

    def test_none_value(self):
        """Test parsing None value."""
        assert normalize_hex_field(None) == b""

    def test_invalid_hex_raises_error(self):
        """Test that invalid hex string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid hex string"):
            normalize_hex_field("0xGGGG")

    def test_unsupported_type_raises_error(self):
        """Test that unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported hex field type"):
            normalize_hex_field(12345)


class TestNormalizeHexString:
    """Test normalize_hex_string function for consistent string output."""

    def test_with_prefix_from_string_without(self):
        """Test adding 0x prefix to string without it."""
        result = normalize_hex_string("1234", with_prefix=True)
        assert result == "0x1234"

    def test_with_prefix_from_string_with(self):
        """Test keeping 0x prefix when already present."""
        result = normalize_hex_string("0x1234", with_prefix=True)
        assert result == "0x1234"

    def test_without_prefix_from_string_with(self):
        """Test removing 0x prefix from string."""
        result = normalize_hex_string("0x1234", with_prefix=False)
        assert result == "1234"

    def test_without_prefix_from_string_without(self):
        """Test string without prefix stays without prefix."""
        result = normalize_hex_string("1234", with_prefix=False)
        assert result == "1234"

    def test_from_hexbytes(self):
        """Test converting HexBytes to string."""
        hex_obj = HexBytes("0x1234")
        result = normalize_hex_string(hex_obj, with_prefix=True)
        assert result == "0x1234"

    def test_from_bytes(self):
        """Test converting raw bytes to string."""
        result = normalize_hex_string(b"\x12\x34", with_prefix=True)
        assert result == "0x1234"


class TestNormalizeWeb3Transaction:
    """Test normalizing Web3 transaction data structures."""

    def test_normalize_simple_transaction(self):
        """Test normalizing a simple transaction dict."""
        tx_data = {
            "hash": HexBytes("0x1234"),
            "data": HexBytes("0xabcd"),
            "blockNumber": 12345,
        }
        result = normalize_web3_transaction(tx_data)

        assert result["hash"] == "0x1234"
        assert result["data"] == "0xabcd"
        assert result["blockNumber"] == 12345

    def test_normalize_transaction_with_logs(self):
        """Test normalizing transaction with nested logs."""
        tx_data = {
            "hash": HexBytes("0x1234"),
            "logs": [
                {
                    "data": HexBytes("0xabcd"),
                    "topics": [HexBytes("0xdead"), HexBytes("0xbeef")],
                },
                {
                    "data": HexBytes("0x5678"),
                    "topics": [HexBytes("0xcafe")],
                },
            ],
        }
        result = normalize_web3_transaction(tx_data)

        assert result["hash"] == "0x1234"
        assert result["logs"][0]["data"] == "0xabcd"
        assert result["logs"][0]["topics"] == ["0xdead", "0xbeef"]
        assert result["logs"][1]["data"] == "0x5678"
        assert result["logs"][1]["topics"] == ["0xcafe"]

    def test_normalize_raw_bytes(self):
        """Test normalizing raw bytes fields."""
        tx_data = {
            "data": b"\x12\x34",
            "hash": b"\xab\xcd",
        }
        result = normalize_web3_transaction(tx_data)

        assert result["data"] == "0x1234"
        assert result["hash"] == "0xabcd"

    def test_normalize_preserves_other_types(self):
        """Test that non-hex types are preserved."""
        tx_data = {
            "blockNumber": 12345,
            "value": "1000000000000000000",
            "status": 1,
            "address": "0x1234567890123456789012345678901234567890",
        }
        result = normalize_web3_transaction(tx_data)

        assert result == tx_data


class TestNormalizeLog:
    """Test normalizing individual log entries."""

    def test_normalize_log_with_prefix(self):
        """Test normalizing log that already has 0x prefix."""
        log = {
            "data": "0x1234",
            "topics": ["0xabcd", "0xef01"],
            "address": "0xToken",
        }
        result = normalize_log(log)

        assert result["data"] == "0x1234"
        assert result["topics"] == ["0xabcd", "0xef01"]
        assert result["address"] == "0xToken"

    def test_normalize_log_without_prefix(self):
        """Test normalizing log without 0x prefix."""
        log = {
            "data": "1234",
            "topics": ["abcd", "ef01"],
            "address": "0xToken",
        }
        result = normalize_log(log)

        assert result["data"] == "0x1234"
        assert result["topics"] == ["0xabcd", "0xef01"]
        assert result["address"] == "0xToken"


class TestParseLogData:
    """Test parse_log_data convenience function."""

    def test_parse_valid_data(self):
        """Test parsing valid log data."""
        log = {"data": "0x1234abcd"}
        result = parse_log_data(log)
        assert result == b"\x12\x34\xab\xcd"

    def test_parse_data_without_prefix(self):
        """Test parsing data without 0x prefix."""
        log = {"data": "1234abcd"}
        result = parse_log_data(log)
        assert result == b"\x12\x34\xab\xcd"

    def test_parse_missing_data_raises_error(self):
        """Test that missing data field raises ValueError."""
        log = {"topics": ["0xabcd"]}
        with pytest.raises(ValueError, match="no 'data' field"):
            parse_log_data(log)


class TestParseLogTopics:
    """Test parse_log_topics convenience function."""

    def test_parse_valid_topics(self):
        """Test parsing valid log topics."""
        log = {"topics": ["0xabcd", "0xef01", "0x1234"]}
        result = parse_log_topics(log)
        assert result == [b"\xab\xcd", b"\xef\x01", b"\x12\x34"]

    def test_parse_topics_without_prefix(self):
        """Test parsing topics without 0x prefix."""
        log = {"topics": ["abcd", "ef01", "1234"]}
        result = parse_log_topics(log)
        assert result == [b"\xab\xcd", b"\xef\x01", b"\x12\x34"]

    def test_parse_missing_topics_raises_error(self):
        """Test that missing topics field raises ValueError."""
        log = {"data": "0xabcd"}
        with pytest.raises(ValueError, match="no 'topics' field"):
            parse_log_topics(log)

    def test_parse_empty_topics(self):
        """Test parsing empty topics list."""
        log = {"topics": []}
        result = parse_log_topics(log)
        assert result == []


class TestRealWorldScenarios:
    """Test with real-world transaction data patterns."""

    def test_uniswap_v2_swap_data(self):
        """Test parsing Uniswap V2 swap data (128 bytes, 4 uint256 values)."""
        # Real Uniswap V2 swap data without 0x prefix (from JSON storage)
        data = "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001a6431c215f37491c400000000000000000000000000000000000000000000000000057c16a62bf16e70000000000000000000000000000000000000000000000000000000000000000"

        result = normalize_hex_field(data)
        assert len(result) == 128  # 4 * 32 bytes

        # Parse amounts
        amount0_in = int.from_bytes(result[0:32], byteorder="big")
        amount1_in = int.from_bytes(result[32:64], byteorder="big")
        amount0_out = int.from_bytes(result[64:96], byteorder="big")
        amount1_out = int.from_bytes(result[96:128], byteorder="big")

        assert amount0_in == 0
        assert amount1_in > 0
        assert amount0_out > 0
        assert amount1_out == 0

    def test_uniswap_v3_swap_data(self):
        """Test parsing Uniswap V3 swap data (160 bytes, 5 values)."""
        # Simulated V3 swap data (160 bytes: 2*int256 + uint160 + uint128 + int24)
        data = "0x" + "00" * 160

        result = normalize_hex_field(data)
        assert len(result) == 160

    def test_erc20_transfer_topics(self):
        """Test parsing ERC-20 Transfer event topics."""
        # Real Transfer event topics from transaction
        topics = [
            "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",  # Transfer signature
            "0000000000000000000000005df6a5ac77f3ce10d9a2bb57543863d1f164dc03",  # from address
            "0000000000000000000000000b85b3000bef3e26e01428d1b525a532ea7513b8",  # to address
        ]

        parsed = [normalize_hex_field(t) for t in topics]
        assert len(parsed[0]) == 32  # Event signature is 32 bytes
        assert len(parsed[1]) == 32  # Address padded to 32 bytes
        assert len(parsed[2]) == 32  # Address padded to 32 bytes

        # Extract addresses from last 20 bytes
        from_addr = parsed[1][-20:].hex()
        to_addr = parsed[2][-20:].hex()

        assert len(from_addr) == 40
        assert len(to_addr) == 40
