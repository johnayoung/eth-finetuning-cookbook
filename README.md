# Ethereum Fine-Tuning Cookbook

An open-source educational repository that teaches developers how to fine-tune language models on blockchain data using QLoRA/LoRA techniques. This cookbook provides complete working examples, scripts, and interactive tutorials for fine-tuning models to extract structured intents from Ethereum transactions.

## 🎯 Project Overview

This cookbook demonstrates end-to-end fine-tuning of large language models (LLMs) using Ethereum transaction analysis as a practical, real-world use case. Learn how to:

- Extract and decode Ethereum transaction data (ETH transfers, ERC-20, Uniswap V2/V3)
- Prepare instruction-tuning datasets from blockchain data
- Fine-tune 7B parameter models on consumer GPUs using QLoRA
- Evaluate model performance with quantitative metrics
- Deploy fine-tuned models for inference

## 🎓 Educational Focus

The primary goal is **teaching fine-tuning techniques** with blockchain data as a concrete application. This is not a production transaction analyzer—it's a hands-on learning resource for:

- ML engineers exploring parameter-efficient fine-tuning (PEFT)
- Blockchain developers interested in AI/ML applications
- Researchers studying domain adaptation of LLMs
- Students learning practical fine-tuning workflows

## 💻 Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA RTX 3060 or equivalent (12GB VRAM)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space (for models and datasets)
- **CUDA**: 11.8 or newer

### Recommended
- **GPU**: RTX 3080/3090 or RTX 4070/4080 (16GB+ VRAM)
- **RAM**: 32GB system memory
- Better hardware = faster training and ability to use larger batch sizes

### Cloud Alternatives
- Google Colab Pro (T4 GPU, 15GB VRAM) - included notebooks work out-of-the-box
- AWS EC2 g5.xlarge or g4dn.xlarge instances
- Lambda Labs / RunPod GPU rentals

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or newer
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Ethereum RPC endpoint (Infura, Alchemy, or local node)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/eth-finetuning-cookbook.git
cd eth-finetuning-cookbook

# Install dependencies
uv pip install -e ".[dev]"
# or with pip: pip install -e ".[dev]"
```

### Usage

All scripts must be run as Python modules using the `-m` flag:

```bash
# Fetch transactions from Ethereum
python -m scripts.extraction.fetch_transactions \
    --rpc-url https://eth.llamarpc.com \
    --tx-hashes data/transactions.txt \
    --output data/raw/transactions.json

# Run tests
pytest tests/ -v
```

**Note:** Do not run scripts directly (e.g., `python scripts/extraction/fetch_transactions.py`). Always use `python -m` to avoid import issues

### Installation

**Using uv (Recommended - Fast & Modern)**
```bash
# Clone the repository
git clone https://github.com/your-org/eth-finetuning-cookbook.git
cd eth-finetuning-cookbook

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (uses uv.lock for reproducible builds)
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

> **Note**: `uv sync` reads the `uv.lock` file to ensure everyone gets the exact same dependency versions for reproducible builds.

**Using pip (Alternative)**
```bash
# Clone the repository
git clone https://github.com/your-org/eth-finetuning-cookbook.git
cd eth-finetuning-cookbook

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## 📚 Documentation

Coming soon! Complete documentation will include:

- **Getting Started Guide**: Installation and setup
- **Data Extraction Guide**: Fetching and decoding transactions
- **Fine-Tuning Guide**: Step-by-step training tutorial
- **Evaluation Guide**: Assessing model performance

## 📓 Interactive Notebooks

Five educational Jupyter notebooks will guide you through the entire pipeline:

1. **Data Exploration**: Understanding Ethereum transaction structure
2. **Data Extraction**: Fetching and decoding transactions
3. **Dataset Preparation**: Creating instruction-tuning datasets
4. **Fine-Tuning**: Training models with QLoRA (Colab-compatible)
5. **Evaluation**: Testing and analyzing model performance

## 🎯 Performance Targets

This cookbook demonstrates achieving:

- **≥90% accuracy** on transaction amounts, addresses, and protocol identification
- **≥60 Flesch Reading Ease** score for generated descriptions
- **<4 hours training time** on RTX 3060 (12GB VRAM)
- **<12GB VRAM usage** during training

## 🛠️ Technology Stack

- **Data**: web3.py, pandas, eth-abi
- **ML/Training**: PyTorch, Hugging Face Transformers, PEFT, bitsandbytes
- **Base Models**: Mistral-7B or Llama-2-7B
- **Development**: pytest, Jupyter, click

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! This is an educational project aimed at helping developers learn fine-tuning techniques. Please feel free to:

- Report issues or bugs
- Suggest improvements to tutorials
- Add support for new transaction types
- Improve documentation and examples

## ⚠️ Disclaimer

This is an educational project for learning fine-tuning techniques. It is not intended for production use as a transaction analysis tool. Always verify transaction data through official block explorers and audit any smart contract interactions.

## 📖 Citation

If you use this cookbook in your research or projects, please cite:

```bibtex
@misc{eth-finetuning-cookbook,
  title={Ethereum Fine-Tuning Cookbook: Educational Guide to LLM Fine-Tuning on Blockchain Data},
  author={Contributors},
  year={2025},
  url={https://github.com/your-org/eth-finetuning-cookbook}
}
```

---

**Status**: 🚧 Under active development (MVP in progress)

**Current Phase**: Project setup and configuration
