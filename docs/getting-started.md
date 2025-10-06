# Getting Started Guide

Welcome to the Ethereum Fine-Tuning Cookbook! This guide will help you set up your environment and get started with fine-tuning language models on blockchain data.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Verify Installation](#verify-installation)
5. [Next Steps](#next-steps)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

#### Minimum Requirements (For Fine-Tuning)
- **GPU**: NVIDIA RTX 3060 or equivalent (12GB VRAM)
- **System RAM**: 16GB
- **Storage**: 50GB free space
- **CUDA**: Version 11.8 or newer

#### For Data Extraction Only
- **CPU**: Any modern CPU (no GPU required)
- **System RAM**: 8GB
- **Storage**: 10GB free space

#### Recommended Configuration
- **GPU**: RTX 3080/3090 or RTX 4070/4080 (16GB+ VRAM)
- **System RAM**: 32GB
- **Storage**: 100GB SSD
- **Internet**: Stable connection for RPC access

### Software Requirements

- **Python**: Version 3.10 or newer
- **Operating System**: Linux (Ubuntu 20.04+), macOS 12+, or Windows 11 with WSL2
- **CUDA Toolkit**: 11.8+ (for GPU training)
- **Git**: For cloning the repository

### External Services

You'll need access to an Ethereum RPC endpoint. Choose one of:

1. **Infura** (Recommended for beginners)
   - Sign up at: https://infura.io
   - Free tier: 100,000 requests/day
   - Get your API key from the dashboard

2. **Alchemy**
   - Sign up at: https://alchemy.com
   - Free tier: 300M compute units/month
   - More generous rate limits than Infura

3. **Local Node** (Advanced)
   - Run your own Geth or Erigon node
   - Best for production use
   - Requires significant storage (>2TB for archive node)

4. **Public RPC Endpoints** (Not recommended for production)
   - Free but rate-limited
   - Examples: Cloudflare, Ankr

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/johnayoung/eth-finetuning-cookbook.git
cd eth-finetuning-cookbook
```

### Step 2: Install uv Package Manager

The `uv` package manager is a fast, modern alternative to pip. Install it:

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (using pip):**
```bash
pip install uv
```

Verify installation:
```bash
uv --version
```

### Step 3: Create Virtual Environment

Using uv (recommended):
```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate     # On Windows
```

Using standard Python venv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate     # On Windows
```

### Step 4: Install Dependencies

Install all dependencies including development tools:

```bash
# Using uv (recommended - much faster)
uv pip install -e ".[dev]"

# OR using standard pip
pip install -e ".[dev]"
```

This installs:
- **Core dependencies**: web3.py, pandas, torch, transformers
- **Fine-tuning tools**: peft, bitsandbytes
- **Development tools**: pytest, jupyter, black
- **Utilities**: click, pyyaml, textstat

**Installation time**: 5-10 minutes depending on internet speed

### Step 5: Verify CUDA Installation (For GPU Training)

Check if PyTorch can access your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3060
```

If CUDA is not available:
1. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
3. Reinstall PyTorch with CUDA support:
   ```bash
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Configuration

### Step 1: Configure RPC Endpoint

Copy the example configuration and add your RPC URL:

```bash
cp configs/extraction_config.yaml.example configs/extraction_config.yaml
# OR create manually if example doesn't exist
```

Edit `configs/extraction_config.yaml`:

```yaml
# Ethereum RPC Configuration
rpc:
  # Replace with your actual RPC endpoint
  url: "https://mainnet.infura.io/v3/YOUR_API_KEY_HERE"
  
  # Rate limiting (requests per second)
  rate_limit:
    requests_per_second: 5
    retry_attempts: 3
    backoff_factor: 2.0

# Data extraction settings
extraction:
  batch_size: 10
  timeout: 30
  save_raw: true
```

**For Infura:**
```yaml
url: "https://mainnet.infura.io/v3/YOUR_INFURA_API_KEY"
```

**For Alchemy:**
```yaml
url: "https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_API_KEY"
```

**For Local Node:**
```yaml
url: "http://localhost:8545"
```

### Step 2: Configure Training Parameters (Optional)

The default training configuration in `configs/training_config.yaml` is optimized for 12GB VRAM. You can adjust if needed:

```yaml
# Model configuration
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"
  
# QLoRA configuration (4-bit quantization)
lora:
  r: 16                              # LoRA rank
  alpha: 32                          # LoRA alpha
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]
  
# Training hyperparameters
training:
  learning_rate: 2.0e-4
  batch_size: 1                      # Micro batch size
  gradient_accumulation_steps: 16    # Effective batch = 16
  max_seq_length: 2048
  num_epochs: 3
  warmup_steps: 100
  
# Memory optimization
optimization:
  gradient_checkpointing: true
  bf16: true                         # Use bfloat16 mixed precision
  optim: "paged_adamw_32bit"
```

**For 16GB VRAM**, you can increase batch size:
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

**For 24GB+ VRAM**, you can use larger models:
```yaml
model:
  base_model: "meta-llama/Llama-2-13b-chat-hf"
training:
  batch_size: 4
  gradient_accumulation_steps: 4
```

### Step 3: Configure Evaluation Metrics (Optional)

Edit `configs/evaluation_config.yaml` to set your quality thresholds:

```yaml
# Evaluation metrics configuration
metrics:
  # Accuracy thresholds
  accuracy:
    overall_target: 0.90           # 90% overall accuracy target
    amount_tolerance: 0.01         # ±1% tolerance for amounts
    
  # Readability target (for text outputs)
  readability:
    flesch_reading_ease_min: 60    # Minimum Flesch Reading Ease score
    
  # Per-protocol requirements
  per_protocol:
    ethereum:
      min_accuracy: 0.95
    erc20:
      min_accuracy: 0.90
    uniswap_v2:
      min_accuracy: 0.85
    uniswap_v3:
      min_accuracy: 0.85
```

## Verify Installation

Run the test suite to ensure everything is working:

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test modules
uv run pytest tests/test_extraction.py -v
uv run pytest tests/test_decoders.py -v
uv run pytest tests/test_dataset.py -v

# Check test coverage
uv run pytest tests/ --cov=src --cov=scripts --cov-report=term-missing
```

Expected output: All tests should pass (green checkmarks).

### Quick Functionality Check

Test the extraction pipeline with sample data:

```bash
# Fetch a sample transaction (requires RPC endpoint configured)
python scripts/fetch_transactions.py \
  --tx-hashes tests/fixtures/sample_tx_hashes.txt \
  --output data/raw/test_fetch.json

# Decode the transactions
python scripts/decode_transactions.py \
  --input data/raw/test_fetch.json \
  --output data/processed/test_decoded.csv
```

If these commands complete without errors, your installation is successful!

## Next Steps

Now that your environment is set up, you can:

1. **Explore the Notebooks**: Start with interactive tutorials
   ```bash
   jupyter notebook notebooks/
   ```
   - `01-data-exploration.ipynb` - Learn about Ethereum transaction structure
   - `02-data-extraction.ipynb` - Fetch and decode transactions
   - `03-dataset-preparation.ipynb` - Prepare training data
   - `04-fine-tuning.ipynb` - Train your model
   - `05-evaluation.ipynb` - Evaluate performance

2. **Extract Your Own Data**: Follow the [Data Extraction Guide](data-extraction-guide.md)

3. **Prepare Training Dataset**: See [Dataset Preparation](data-extraction-guide.md#dataset-preparation)

4. **Fine-Tune a Model**: Follow the [Fine-Tuning Guide](fine-tuning-guide.md)

5. **Evaluate Results**: Read the [Evaluation Guide](evaluation-guide.md)

## Troubleshooting

### Common Issues

#### 1. `uv: command not found`

**Solution**: Add uv to your PATH or restart your terminal after installation.

```bash
# Check if uv is installed
which uv

# If not found, reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
```

#### 2. CUDA Not Available

**Problem**: PyTorch doesn't detect your GPU.

**Solution**:
1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch with CUDA support:
   ```bash
   uv pip uninstall torch torchvision torchaudio
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Check CUDA version compatibility:
   - PyTorch 2.0+ requires CUDA 11.8+
   - Use `nvidia-smi` to check your CUDA version

#### 3. Out of Memory During Training

**Problem**: CUDA out of memory error.

**Solution**:
1. Reduce batch size in `configs/training_config.yaml`:
   ```yaml
   training:
     batch_size: 1
     gradient_accumulation_steps: 16
   ```

2. Reduce sequence length:
   ```yaml
   training:
     max_seq_length: 1024  # Down from 2048
   ```

3. Enable gradient checkpointing (should be on by default):
   ```yaml
   optimization:
     gradient_checkpointing: true
   ```

4. Close other GPU applications (browsers, other Python processes)

#### 4. RPC Rate Limiting

**Problem**: `Too Many Requests` or `429` errors.

**Solution**:
1. Increase delay between requests in `configs/extraction_config.yaml`:
   ```yaml
   rate_limit:
     requests_per_second: 2  # Slower rate
     retry_attempts: 5
     backoff_factor: 3.0
   ```

2. Use a paid RPC plan with higher limits

3. Run extraction in smaller batches:
   ```bash
   python scripts/fetch_transactions.py \
     --tx-hashes sample_hashes.txt \
     --batch-size 5 \
     --output data/raw/batch.json
   ```

#### 5. Import Errors

**Problem**: `ModuleNotFoundError` or import errors.

**Solution**:
1. Ensure you're in the virtual environment:
   ```bash
   which python
   # Should show: /path/to/eth-finetuning-cookbook/.venv/bin/python
   ```

2. Reinstall in editable mode:
   ```bash
   uv pip install -e ".[dev]"
   ```

3. Check installation:
   ```bash
   uv pip list | grep eth-finetuning
   ```

#### 6. Jupyter Kernel Not Found

**Problem**: Jupyter can't find the project kernel.

**Solution**:
1. Install ipykernel in your virtual environment:
   ```bash
   uv pip install ipykernel
   python -m ipykernel install --user --name=eth-finetuning
   ```

2. Select the kernel in Jupyter:
   - Open notebook
   - Click: Kernel → Change Kernel → eth-finetuning

#### 7. Test Failures

**Problem**: Tests fail during verification.

**Solution**:
1. Check if fixture files exist:
   ```bash
   ls -la tests/fixtures/
   ```

2. Regenerate fixtures if missing:
   ```bash
   # This requires RPC access
   python scripts/fetch_transactions.py \
     --tx-hashes tests/fixtures/sample_tx_hashes.txt \
     --output tests/fixtures/sample_transactions.json
   ```

3. Run tests with verbose output:
   ```bash
   uv run pytest tests/ -vv --tb=short
   ```

### Getting Help

If you encounter issues not covered here:

1. **Check the Documentation**:
   - [Data Extraction Guide](data-extraction-guide.md)
   - [Fine-Tuning Guide](fine-tuning-guide.md)
   - [Evaluation Guide](evaluation-guide.md)

2. **Search Existing Issues**:
   - GitHub Issues: https://github.com/johnayoung/eth-finetuning-cookbook/issues

3. **Open a New Issue**:
   - Provide your system info:
     ```bash
     uv run python -c "import torch; import sys; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
     ```
   - Include error messages and logs
   - Describe steps to reproduce

4. **Community Support**:
   - GitHub Discussions
   - Stack Overflow (tag: `ethereum` + `pytorch`)

## Additional Resources

- **Project README**: [README.md](../README.md)
- **Technical Specification**: [SPEC.md](SPEC.md)
- **Implementation Roadmap**: [ROADMAP.md](ROADMAP.md)
- **Hugging Face PEFT Docs**: https://huggingface.co/docs/peft
- **Web3.py Documentation**: https://web3py.readthedocs.io
- **PyTorch Tutorials**: https://pytorch.org/tutorials

---

**Next**: [Data Extraction Guide](data-extraction-guide.md) →
