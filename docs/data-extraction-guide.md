# Data Extraction Guide

This guide explains how to extract and decode Ethereum transaction data for training your fine-tuned model.

## Table of Contents

1. [Overview](#overview)
2. [Transaction Fetching](#transaction-fetching)
3. [Transaction Decoding](#transaction-decoding)
4. [Supported Transaction Types](#supported-transaction-types)
5. [Dataset Preparation](#dataset-preparation)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Overview

The data extraction pipeline consists of three main stages:

```
1. Fetch Transactions → 2. Decode Transactions → 3. Prepare Dataset
   (Raw blockchain data)    (Structured intents)     (Training format)
```

### What You'll Need

- Ethereum RPC endpoint (Infura, Alchemy, or local node)
- Transaction hashes to analyze (or block range)
- Configured `configs/extraction_config.yaml`

### Pipeline Architecture

```
┌─────────────────┐
│  RPC Endpoint   │
│  (Ethereum Node)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fetch           │
│ Transactions    │  → data/raw/transactions.json
│ (with receipts) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Decode          │
│ Transaction     │  → data/processed/decoded.csv
│ Types           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract         │
│ Intents         │  → data/datasets/train.jsonl
│ & Format        │
└─────────────────┘
```

## Transaction Fetching

### Basic Usage

Fetch transactions from a list of hashes:

```bash
python scripts/fetch_transactions.py \
  --tx-hashes data/tx_hashes.txt \
  --output data/raw/transactions.json
```

### Input Format

Create a text file with transaction hashes (one per line):

**`data/tx_hashes.txt`:**
```
0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890
0x9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba
```

### Advanced Options

```bash
python scripts/fetch_transactions.py \
  --tx-hashes data/tx_hashes.txt \
  --output data/raw/transactions.json \
  --rpc-url https://mainnet.infura.io/v3/YOUR_KEY \
  --batch-size 10 \
  --max-workers 4 \
  --retry-attempts 3
```

**Parameters:**
- `--tx-hashes`: Path to file with transaction hashes
- `--output`: Where to save the raw transaction data (JSON)
- `--rpc-url`: Ethereum RPC endpoint (defaults to config file)
- `--batch-size`: Number of transactions to fetch in parallel (default: 10)
- `--max-workers`: Number of concurrent workers (default: 4)
- `--retry-attempts`: Number of retries for failed requests (default: 3)

### Output Format

The fetcher saves raw transaction data in JSON format:

```json
[
  {
    "hash": "0x1234...abcd",
    "block_number": 18500000,
    "from": "0xSender...",
    "to": "0xReceiver...",
    "value": "1000000000000000000",
    "input": "0xa9059cbb...",
    "gas": 21000,
    "gas_price": "30000000000",
    "nonce": 42,
    "status": 1,
    "logs": [
      {
        "address": "0xToken...",
        "topics": ["0xddf252ad..."],
        "data": "0x..."
      }
    ]
  }
]
```

### Finding Transaction Hashes

#### Method 1: Etherscan

1. Go to https://etherscan.io
2. Search for an address or token
3. Click on "Transactions" tab
4. Copy transaction hashes

#### Method 2: Block Explorer API

```python
import requests

# Get recent transactions for an address
address = "0x..."
url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=desc&apikey=YOUR_API_KEY"

response = requests.get(url)
txs = response.json()['result']

# Save hashes to file
with open('data/tx_hashes.txt', 'w') as f:
    for tx in txs[:100]:  # First 100 transactions
        f.write(tx['hash'] + '\n')
```

#### Method 3: Filter by Protocol

Use the notebooks to filter transactions by protocol:

```python
# In Jupyter notebook
from web3 import Web3

w3 = Web3(Web3.HTTPProvider('YOUR_RPC_URL'))

# Get transactions from recent blocks
block = w3.eth.get_block('latest', full_transactions=True)
for tx in block.transactions:
    # Filter for Uniswap V2 Router
    if tx['to'] == '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D':
        print(tx['hash'].hex())
```

## Transaction Decoding

### Basic Usage

Decode fetched transactions:

```bash
python scripts/decode_transactions.py \
  --input data/raw/transactions.json \
  --output data/processed/decoded.csv
```

### With RPC (for token metadata)

Some decoders need RPC access to fetch token metadata (symbols, decimals):

```bash
python scripts/decode_transactions.py \
  --input data/raw/transactions.json \
  --output data/processed/decoded.csv \
  --rpc-url https://mainnet.infura.io/v3/YOUR_KEY
```

### Output Format

The decoder creates a CSV with structured transaction data:

| tx_hash  | block    | timestamp  | from     | to            | value               | decoded_action | protocol   | assets    | amounts  |
| -------- | -------- | ---------- | -------- | ------------- | ------------------- | -------------- | ---------- | --------- | -------- |
| 0x123... | 18500000 | 1697200000 | 0xABC... | 0xDEF...      | 1000000000000000000 | transfer       | ethereum   | ETH       | 1.0      |
| 0x456... | 18500001 | 1697200012 | 0xGHI... | 0xJKL...      | 0                   | transfer       | erc20      | USDC      | 1000.0   |
| 0x789... | 18500002 | 1697200024 | 0xMNO... | 0xUniV2Router | 0                   | swap           | uniswap_v2 | USDC→WETH | 1000→0.5 |

### Decoded Fields

- **tx_hash**: Transaction hash
- **block**: Block number
- **timestamp**: Unix timestamp
- **from**: Sender address (checksummed)
- **to**: Receiver address (checksummed)
- **value**: ETH value in wei
- **decoded_action**: Action type (transfer, swap, etc.)
- **protocol**: Protocol identifier (ethereum, erc20, uniswap_v2, uniswap_v3)
- **assets**: Asset symbols involved
- **amounts**: Amount values

## Supported Transaction Types

### 1. ETH Transfers

**Description**: Native ETH transfers between addresses.

**Example:**
```python
{
  "action": "transfer",
  "protocol": "ethereum",
  "from": "0xSender...",
  "to": "0xReceiver...",
  "amount_wei": "1000000000000000000",
  "amount_eth": 1.0,
  "status": "success"
}
```

**Identification**: Transaction with non-zero `value` field and empty or minimal `input` data.

### 2. ERC-20 Transfers

**Description**: Token transfers following the ERC-20 standard.

**Example:**
```python
{
  "action": "transfer",
  "protocol": "erc20",
  "token_address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
  "token_symbol": "USDC",
  "from": "0xSender...",
  "to": "0xReceiver...",
  "amount": "1000000000",  # 1000 USDC (6 decimals)
  "decimals": 6
}
```

**Identification**: 
- Event signature: `Transfer(address,address,uint256)`
- Topic 0: `0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef`

**Supported Methods:**
- `transfer(address,uint256)`
- `transferFrom(address,address,uint256)`

### 3. Uniswap V2 Swaps

**Description**: Token swaps on Uniswap V2 AMM pools.

**Example:**
```python
{
  "action": "swap",
  "protocol": "uniswap_v2",
  "pool": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
  "token_in": "USDC",
  "token_out": "WETH",
  "amount_in": "1000000000",
  "amount_out": "500000000000000000",
  "sender": "0xSender...",
  "recipient": "0xRecipient..."
}
```

**Identification**:
- Event signature: `Swap(address,uint256,uint256,uint256,uint256,address)`
- Pools created by Uniswap V2 Factory
- Uses `swapExactTokensForTokens`, `swapTokensForExactTokens`, etc.

**Multi-Hop Swaps**: Parsed as sequential swap operations.

### 4. Uniswap V3 Swaps

**Description**: Token swaps on Uniswap V3 concentrated liquidity pools.

**Example:**
```python
{
  "action": "swap",
  "protocol": "uniswap_v3",
  "pool": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
  "token_in": "USDC",
  "token_out": "WETH",
  "amount_in": "1000000000",
  "amount_out": "500000000000000000",
  "sqrt_price_x96": "1234567890...",
  "liquidity": "9876543210...",
  "tick": 201234
}
```

**Identification**:
- Event signature: `Swap(address,address,int256,int256,uint160,uint128,int24)`
- Pools created by Uniswap V3 Factory
- Includes tick and sqrtPriceX96 information

## Dataset Preparation

After decoding, prepare the data for training:

```bash
python scripts/dataset/prepare_training_data.py \
  --input data/processed/decoded.csv \
  --output data/datasets/ \
  --split 0.7 0.15 0.15
```

### Split Ratios

- **70% Training**: Used to train the model
- **15% Validation**: Used during training for evaluation
- **15% Test**: Held out for final evaluation

### Output Files

- `data/datasets/train.jsonl`: Training dataset
- `data/datasets/validation.jsonl`: Validation dataset
- `data/datasets/test.jsonl`: Test dataset

### Format

Each line is a JSON object with instruction-tuning format:

```json
{
  "instruction": "Extract the structured intent from this Ethereum transaction:",
  "input": "{\"tx_hash\": \"0x123...\", \"from\": \"0xABC...\", \"to\": \"0xDEF...\", \"value\": \"1000000000000000000\", \"protocol\": \"ethereum\"}",
  "output": "{\"action\": \"transfer\", \"protocol\": \"ethereum\", \"assets\": [\"ETH\"], \"amounts\": [\"1.0\"], \"outcome\": \"success\"}"
}
```

## Troubleshooting

### Issue 1: RPC Rate Limiting

**Symptoms:**
- `HTTPError: 429 Too Many Requests`
- `Exceeded rate limit` errors
- Slow or stalled fetching

**Solutions:**

1. **Reduce request rate:**
   ```yaml
   # configs/extraction_config.yaml
   rate_limit:
     requests_per_second: 2  # Slower
     retry_attempts: 5
     backoff_factor: 3.0
   ```

2. **Use batch fetching:**
   ```bash
   python scripts/fetch_transactions.py \
     --batch-size 5 \  # Smaller batches
     --max-workers 2   # Fewer concurrent requests
   ```

3. **Upgrade RPC plan:**
   - Infura: Growth plan (1M requests/month)
   - Alchemy: Growth plan (unlimited compute units)

4. **Use multiple endpoints:**
   ```python
   # Round-robin between endpoints
   rpc_urls = [
       "https://mainnet.infura.io/v3/KEY1",
       "https://eth-mainnet.g.alchemy.com/v2/KEY2"
   ]
   ```

### Issue 2: Transaction Not Found

**Symptoms:**
- `Transaction not found` errors
- `None` returned for transaction data

**Solutions:**

1. **Verify transaction hash:**
   - Check on Etherscan: https://etherscan.io/tx/HASH
   - Ensure hash is complete and correct

2. **Check network:**
   - Are you querying mainnet or testnet?
   - Use correct RPC endpoint for the network

3. **Wait for confirmation:**
   - Recent transactions might not be fully propagated
   - Wait a few blocks for confirmation

### Issue 3: Decoder Fails on Transaction

**Symptoms:**
- `Unsupported transaction type`
- Missing fields in decoded output
- Decoder returns `None`

**Solutions:**

1. **Check transaction type:**
   ```python
   # Inspect transaction manually
   from web3 import Web3
   w3 = Web3(Web3.HTTPProvider('YOUR_RPC_URL'))
   tx = w3.eth.get_transaction('0x...')
   print(tx)
   ```

2. **Verify it's a supported type:**
   - ETH transfer: `value > 0`, minimal `input`
   - ERC-20: Look for Transfer event in logs
   - Uniswap V2/V3: Check pool address and Swap event

3. **Add custom decoder:**
   - See `src/eth_finetuning/extraction/decoders/` for examples
   - Implement decoder for new protocol

### Issue 4: Token Metadata Not Found

**Symptoms:**
- Token symbol shows as `UNKNOWN`
- Decimals default to 18
- Warning: `Could not fetch token metadata`

**Solutions:**

1. **Provide RPC URL:**
   ```bash
   python scripts/decode_transactions.py \
     --rpc-url https://mainnet.infura.io/v3/YOUR_KEY
   ```

2. **Manual metadata cache:**
   Create `data/token_metadata.json`:
   ```json
   {
     "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {
       "symbol": "USDC",
       "decimals": 6
     }
   }
   ```

3. **Use token list:**
   - Download: https://tokenlists.org
   - Pre-cache common tokens

### Issue 5: Out of Memory During Processing

**Symptoms:**
- `MemoryError` during decoding
- System freezes or swaps
- Killed by OOM killer

**Solutions:**

1. **Process in smaller batches:**
   ```bash
   # Split input file
   split -l 100 data/tx_hashes.txt data/batch_
   
   # Process each batch
   for batch in data/batch_*; do
     python scripts/fetch_transactions.py \
       --tx-hashes $batch \
       --output data/raw/$(basename $batch).json
   done
   ```

2. **Use streaming processing:**
   ```python
   # Process one transaction at a time
   # Modify scripts to use generators instead of loading all into memory
   ```

3. **Increase system swap:**
   ```bash
   # Linux: Add swap space
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

## Best Practices

### 1. Data Quality

✅ **DO:**
- Verify transactions exist before adding to dataset
- Include mix of different transaction types
- Balance protocol representation (ETH, ERC-20, Uniswap)
- Filter out failed transactions or keep them labeled
- Validate all addresses are checksummed

❌ **DON'T:**
- Use only one transaction type
- Include spam or dust transactions
- Mix mainnet and testnet transactions
- Ignore transaction status

### 2. Dataset Size

**Minimum for fine-tuning:** 500-1,000 transactions
**Recommended:** 5,000-10,000 transactions
**Optimal:** 50,000+ transactions

**Distribution:**
- ETH transfers: 30-40%
- ERC-20 transfers: 30-40%
- Uniswap V2: 10-15%
- Uniswap V3: 10-15%

### 3. RPC Usage

✅ **DO:**
- Cache fetched data (don't re-fetch)
- Use batch requests when possible
- Implement exponential backoff
- Monitor your API usage
- Use paid plans for production

❌ **DON'T:**
- Hammer free endpoints
- Ignore rate limits
- Make synchronous requests for large datasets
- Store API keys in code

### 4. Data Organization

**Recommended structure:**
```
data/
├── raw/                      # Raw fetched data
│   ├── batch_001.json
│   ├── batch_002.json
│   └── ...
├── processed/                # Decoded data
│   ├── decoded_001.csv
│   ├── decoded_002.csv
│   └── combined.csv
└── datasets/                 # Training format
    ├── train.jsonl
    ├── validation.jsonl
    └── test.jsonl
```

### 5. Version Control

✅ **DO:**
- Tag dataset versions
- Document data sources
- Keep extraction logs
- Save configuration used

**Example:**
```bash
# Tag dataset version
cp data/datasets/train.jsonl data/datasets/train_v1.0.jsonl

# Document
echo "Dataset v1.0 - 10,000 transactions from blocks 18,000,000-18,500,000" > data/datasets/VERSION.txt
```

### 6. Validation

Always validate extracted data:

```bash
# Check row counts
wc -l data/datasets/train.jsonl
wc -l data/datasets/validation.jsonl
wc -l data/datasets/test.jsonl

# Validate JSON format
python -c "import json; [json.loads(line) for line in open('data/datasets/train.jsonl')]"

# Check for nulls
grep -c "null" data/processed/decoded.csv
```

## Advanced Topics

### Custom Decoder Development

Create a decoder for a new protocol:

```python
# src/eth_finetuning/extraction/decoders/aave.py

from typing import Dict, Optional
from web3 import Web3

def decode_aave_deposit(tx: Dict, receipt: Dict, web3: Web3) -> Optional[Dict]:
    """Decode Aave deposit transaction."""
    
    # Check if transaction is for Aave
    aave_pool = "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"
    if tx.get("to", "").lower() != aave_pool.lower():
        return None
    
    # Parse Deposit event from logs
    deposit_event_sig = Web3.keccak(text="Deposit(address,address,uint256,uint256)").hex()
    
    for log in receipt.get("logs", []):
        if log["topics"][0].hex() == deposit_event_sig:
            return {
                "action": "deposit",
                "protocol": "aave_v2",
                "asset": _get_token_symbol(log["topics"][1], web3),
                "amount": int(log["data"], 16),
                "user": Web3.to_checksum_address(log["topics"][2][-40:])
            }
    
    return None
```

### Parallel Processing

Speed up extraction with parallel processing:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_parallel(tx_hashes: List[str], rpc_url: str, max_workers: int = 10):
    """Fetch transactions in parallel."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(w3.eth.get_transaction, tx_hash): tx_hash 
                   for tx_hash in tx_hashes}
        
        for future in as_completed(futures):
            try:
                tx = future.result()
                results.append(dict(tx))
            except Exception as e:
                print(f"Error fetching {futures[future]}: {e}")
    
    return results
```

### Blockchain Data Sources

Beyond RPC endpoints:

1. **The Graph**: Query indexed blockchain data
   - https://thegraph.com
   - GraphQL API for efficient queries

2. **Dune Analytics**: Export curated datasets
   - https://dune.com
   - Pre-filtered transaction sets

3. **BigQuery**: Ethereum public dataset
   - Google Cloud BigQuery
   - SQL queries on full blockchain history

## Next Steps

- **Fine-Tune Your Model**: [Fine-Tuning Guide](fine-tuning-guide.md)
- **Evaluate Performance**: [Evaluation Guide](evaluation-guide.md)
- **Explore Notebooks**: `notebooks/02-data-extraction.ipynb`

---

**Previous**: [Getting Started](getting-started.md) | **Next**: [Fine-Tuning Guide](fine-tuning-guide.md) →
