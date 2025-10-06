# MVP Technical Specification: Ethereum Fine-Tuning Cookbook

## Project Type
**Educational Cookbook** - An open-source educational repository that teaches developers how to fine-tune language models on blockchain data. It provides complete working examples, scripts, and tutorials for QLoRA/LoRA fine-tuning, using Ethereum transaction analysis as the practical demonstration case. The primary goal is **education about fine-tuning techniques**, with the transaction analysis tool serving as a real-world application to make learning concrete.

## Core Requirements (from Brief)

### MVP Scope
1. **Transaction Intent Extraction**: Extract structured intents from Ethereum transactions (action, assets, protocol, outcome)
2. **Essential Transaction Types**: Decode ETH transfers, ERC-20 operations, and Uniswap V2/V3 swaps
3. **Data Pipeline**: Extensible recipes for extracting training data from transaction receipts and logs
4. **Complete Fine-Tuning Pipeline**: End-to-end workflow runnable on 16GB consumer GPUs (RTX 3060 target)
5. **Performance Target**: Complete fine-tuning process in under 4 hours on 12GB VRAM GPU
6. **Quality Metrics**: 
   - 90% accuracy on transaction amounts, addresses, and protocol identification
   - 60+ Flesch Reading Ease score for generated descriptions

### Post-MVP (Explicitly Out of Scope for V1)
- Multiple output formats (semantic summaries, narratives, flow descriptions)
- Multi-step DeFi operations (flash loans, arbitrage)
- Transaction failure diagnosis from revert data
- Value flow tracing through internal transfers and multicalls

## Technology Stack

### Data Collection & Processing
- **Python 3.10+**: Primary language for data pipelines and ML workflows
- **web3.py**: Ethereum node interaction and transaction decoding
- **pandas**: Data manipulation and CSV/Parquet handling
- **eth-abi**: ABI encoding/decoding for smart contract interactions

### Machine Learning & Fine-Tuning
- **PyTorch 2.0+**: Deep learning framework
- **Hugging Face Transformers**: Pre-trained models and training utilities
- **Hugging Face PEFT (Parameter-Efficient Fine-Tuning)**: QLoRA/LoRA implementation
- **bitsandbytes**: 4-bit quantization for memory efficiency
- **Hugging Face Datasets**: Dataset loading and processing
- **Base Model**: Mistral-7B or Llama-2-7B (balance of performance and VRAM constraints)

### Development & Deployment
- **uv**: Modern Python package manager for fast, reliable dependency management
- **Jupyter Notebooks**: Interactive tutorials and examples
- **pytest**: Testing framework for data pipelines
- **Python click or argparse**: CLI interface for data extraction and training
- **Git LFS**: Version control for large model checkpoints (optional for MVP)

### Infrastructure Requirements
- **CUDA 11.8+ / ROCm**: GPU acceleration
- **12-16GB VRAM**: Target hardware constraint
- **Node Access**: Ethereum RPC endpoint (Infura, Alchemy, or local node)

**Stack Justification:**
- Hugging Face ecosystem: Industry-standard with extensive documentation and community support
- QLoRA/PEFT: Enables fine-tuning 7B models on consumer GPUs through 4-bit quantization
- Python: Dominant language for both blockchain tooling (web3.py) and ML workflows
- Mistral/Llama-2: Open-source models with strong instruction-following capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ETHEREUM NODE (RPC)                          │
│                  (Infura/Alchemy/Local Node)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ web3.py queries
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               DATA EXTRACTION PIPELINE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Transaction  │  │  Receipt &   │  │   Decoder    │          │
│  │   Fetcher    │─▶│ Log Parser   │─▶│  (ETH, ERC20,│          │
│  │              │  │              │  │   Uniswap)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Raw structured data
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              DATASET PREPARATION                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Intent     │  │  Prompt      │  │  HuggingFace │          │
│  │ Extractor    │─▶│ Formatter    │─▶│   Dataset    │          │
│  │ (JSON)       │  │ (Inst+Output)│  │   (train.jsonl│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Training dataset
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              FINE-TUNING PIPELINE                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Base Model   │  │  QLoRA       │  │  Trainer     │          │
│  │ Loader       │─▶│ Config       │─▶│ (HF Trainer) │          │
│  │ (4-bit quant)│  │ (PEFT)       │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Fine-tuned adapter
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              INFERENCE & EVALUATION                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Model +     │  │  Test Set    │  │  Metrics     │          │
│  │  Adapter     │─▶│  Inference   │─▶│ Calculator   │          │
│  │              │  │              │  │ (Accuracy,   │          │
│  │              │  │              │  │  Readability)│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### End-to-End User Flow (MVP)

1. **Setup**: User configures Ethereum RPC endpoint and target transaction hashes
2. **Data Extraction**: Pipeline fetches transactions, decodes transfers/swaps, outputs structured JSON
3. **Dataset Preparation**: Scripts convert raw data to instruction-tuning format (input: tx data → output: structured intent)
4. **Fine-Tuning**: User runs training script with QLoRA configuration, monitors loss over ~3-4 hours
5. **Evaluation**: Model generates intents for test set, metrics calculated (accuracy, readability)
6. **Usage**: Developer loads fine-tuned adapter to analyze new transactions programmatically

**Sample Transaction Processing Flow:**
```
Raw TX Hash
    ↓
web3.getTransaction() → {from, to, value, input, ...}
web3.getTransactionReceipt() → {logs: [...]}
    ↓
Decoder identifies: Uniswap V2 swap event
    ↓
Extract: {action: "swap", protocol: "Uniswap V2", assets: ["USDC", "WETH"], amounts: [1000, 0.5]}
    ↓
Format as prompt: "Analyze this transaction: [TX_DATA]"
Target output: {"action": "swap", "protocol": "uniswap_v2", "input_asset": "USDC", ...}
    ↓
Add to training dataset (train.jsonl)
```

## System Components

### 1. Data Extraction Pipeline
**Purpose:** Fetch and decode Ethereum transactions into structured training data for MVP transaction types

**Inputs:** 
- Ethereum RPC endpoint URL
- List of transaction hashes or block range
- ABI definitions for ERC-20, Uniswap V2/V3

**Outputs:** 
- `decoded.json`: Complete raw transaction data with all fields (preferred for training)
- `decoded.csv`: Human-readable summary with columns: [tx_hash, block, from, to, value, action, protocol, assets, amounts, status]
- Both formats generated simultaneously for different use cases

**Dependencies:** 
- web3.py (Ethereum node access)
- eth-abi (contract decoding)
- pandas (data structuring)

**Key Responsibilities:**
- Connect to Ethereum node via RPC
- Fetch transaction data and receipts in batches
- Decode ETH transfers from transaction value
- Decode ERC-20 Transfer events from logs (using standard ERC-20 ABI)
- Decode Uniswap V2 Swap events (identify pools, extract token pairs, amounts)
- Decode Uniswap V3 Swap events (handle tick-based pricing)
- Handle RPC rate limits and retries
- **Post-MVP**: Multi-call decoding, internal transaction tracing, flash loan detection

### 2. Dataset Preparation Module
**Purpose:** Transform raw transaction data into instruction-tuning format compatible with HuggingFace Trainer

**Inputs:** 
- Decoded transaction JSON from extraction pipeline (preferred; preserves all raw data)
- Alternative: CSV for simple pipelines (may lose some metadata)
- Prompt template configuration
- Train/validation split ratio (e.g., 80/20)

**Outputs:** 
- `train.jsonl`: Training dataset in format `{"instruction": "...", "input": "...", "output": "..."}`
- `validation.jsonl`: Validation dataset
- `test.jsonl`: Held-out test set for final evaluation

**Dependencies:** 
- pandas (data manipulation)
- HuggingFace datasets library
- JSON serialization

**Key Responsibilities:**
- Extract intent structure: {action, assets, protocol, outcome}
- Generate instruction prompts (e.g., "Extract the structured intent from this Ethereum transaction:")
- Format transaction data as model input (JSON or natural language description)
- Create target outputs (structured JSON intents)
- Split data into train/val/test sets (70/15/15 or 80/10/10)
- Validate data quality (no null values in critical fields, valid addresses)
- **Post-MVP**: Multi-format outputs (narrative, flow diagram), failure diagnosis labels

### 3. Fine-Tuning Pipeline
**Purpose:** Execute QLoRA fine-tuning on consumer GPU to create domain-adapted model

**Inputs:** 
- Base model name:
  - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (ungated, 2GB, good for testing)
  - "mistralai/Mistral-7B-Instruct-v0.2" (gated, requires HuggingFace auth, production)
  - "meta-llama/Llama-2-7b-hf" (gated, requires HuggingFace auth)
- Training dataset (train.jsonl)
- Hyperparameters: learning rate, batch size, epochs, LoRA rank
- QLoRA configuration: 4-bit quantization settings

**Outputs:** 
- Fine-tuned LoRA adapter weights (saved to `./output/adapter_model`)
- Training logs (loss, learning rate schedule)
- Checkpoint models at intervals

**Dependencies:** 
- PyTorch
- Transformers (Trainer API, AutoModelForCausalLM)
- PEFT (LoraConfig, get_peft_model)
- bitsandbytes (4-bit quantization)
- accelerate (distributed training utilities)

**Key Responsibilities:**
- Load base model in 4-bit quantized format (reduces 7B model from ~28GB to ~7GB)
- Configure QLoRA parameters:
  - LoRA rank: 8-16 (balance between expressiveness and memory)
  - Target modules: query/value projection layers
  - Alpha scaling: typically 32
- Set training hyperparameters for 12GB VRAM constraint:
  - Batch size: 1-2 with gradient accumulation (8-16 steps)
  - Learning rate: 2e-4 (standard for QLoRA)
  - Epochs: 3-5
  - Max sequence length: 2048 tokens
- Use HuggingFace Trainer with gradient checkpointing enabled
- Save adapters periodically to prevent data loss
- **Post-MVP**: Multi-GPU support, experiment tracking (Weights & Biases), hyperparameter tuning

### 4. Evaluation Module
**Purpose:** Quantify model performance against success metrics from brief

**Inputs:** 
- Fine-tuned model + adapter
- Test dataset (test.jsonl)
- Ground truth intents

**Outputs:** 
- Accuracy report: amounts (% exact match), addresses (% exact match), protocols (% exact match)
- Readability scores: Flesch Reading Ease for generated text
- Confusion matrix for protocol classification
- Sample predictions (JSON)

**Dependencies:** 
- Transformers (model inference)
- scikit-learn (metrics)
- textstat (readability scoring)

**Key Responsibilities:**
- Load fine-tuned model and run inference on test set
- Parse generated JSON intents (handle malformed outputs gracefully)
- Calculate accuracy metrics:
  - **Amount accuracy**: Compare extracted numeric values (allow ±1% tolerance for floating point)
  - **Address accuracy**: Exact string match for Ethereum addresses
  - **Protocol accuracy**: Classification accuracy (ETH, ERC-20, Uniswap V2, Uniswap V3)
- Calculate Flesch Reading Ease score if model generates natural language descriptions
- Generate per-protocol performance breakdown
- **Post-MVP**: Semantic similarity metrics, failure diagnosis accuracy, multi-format evaluation

### 5. Inference Examples
**Purpose:** Demonstrate how to use the fine-tuned model through example scripts (educational reference code, not production API)

**Inputs:** 
- Transaction hash or raw transaction data
- Fine-tuned model path

**Outputs:** 
- Structured intent JSON: `{"action": "...", "assets": [...], "protocol": "...", "outcome": "..."}`

**Dependencies:** 
- Transformers (model loading and inference)
- web3.py (optional: for fetching live transaction data)

**Key Responsibilities:**
- Provide example code showing how to load model + adapter
- Demonstrate inference on sample transactions
- Show how to parse model outputs into structured JSON
- Include error handling patterns as learning examples
- **Post-MVP**: Users can adapt these examples for their production use cases

## File Structure

```
eth-finetuning-cookbook/
├── README.md                          # Quick start, overview, hardware requirements
├── LICENSE                            # Open source license (MIT recommended)
├── pyproject.toml                     # Python project metadata and dependencies
│
├── src/                               # Source package (installable via pip/uv)
│   └── eth_finetuning/
│       ├── __init__.py
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── core/
│       │   │   ├── __init__.py
│       │   │   ├── utils.py           # Web3 connection, retry logic, ABI loading
│       │   │   └── fetcher.py         # Transaction fetching logic
│       │   ├── decoders/
│       │   │   ├── __init__.py
│       │   │   ├── eth.py             # ETH transfer decoder
│       │   │   ├── erc20.py           # ERC-20 token decoder
│       │   │   └── uniswap/           # Uniswap decoders (future)
│       │   │       ├── __init__.py
│       │   │       ├── v2.py
│       │   │       └── v3.py
│       │   ├── abis/
│       │   │   ├── erc20.json
│       │   │   └── ...                # Protocol ABIs
│       │   └── export.py              # CSV/data export utilities
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── preparation.py         # Dataset preparation logic
│       │   ├── intent_extraction.py   # Intent extraction
│       │   └── templates.py           # Prompt templates
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py             # Training logic
│       │   └── config.py              # Training configuration
│       └── evaluation/
│           ├── __init__.py
│           ├── evaluator.py           # Evaluation logic
│           └── metrics.py             # Metrics calculation
│
├── scripts/                           # CLI entry points (thin wrappers)
│   ├── fetch_transactions.py          # CLI: Fetch transactions from Ethereum node
│   ├── decode_transactions.py         # CLI: Decode transactions
│   ├── dataset/
│   │   └── prepare_training_data.py   # CLI: Prepare training dataset
│   ├── training/
│   │   └── train_model.py             # CLI: Run fine-tuning
│   ├── evaluation/
│   │   └── evaluate_model.py          # CLI: Run evaluation
│   └── examples/
│       ├── run_inference.py           # Example: Load model and generate intents
│       └── analyze_transaction.py     # Example: Full pipeline for single transaction
│
├── docs/
│   ├── BRIEF.md                       # Project brief
│   ├── SPEC.md                        # This technical specification
│   ├── ROADMAP.md                     # Implementation roadmap
│   ├── getting-started.md             # Installation and setup guide
│   ├── data-extraction-guide.md       # How to fetch and decode transactions
│   ├── fine-tuning-guide.md           # Step-by-step fine-tuning tutorial
│   └── evaluation-guide.md            # How to assess model performance
│
├── notebooks/
│   ├── 01-data-exploration.ipynb      # Explore Ethereum transaction data
│   ├── 02-data-extraction.ipynb       # Interactive data extraction walkthrough
│   ├── 03-dataset-preparation.ipynb   # Build training dataset
│   ├── 04-fine-tuning.ipynb           # Fine-tuning workflow (can run in Colab)
│   └── 05-evaluation.ipynb            # Evaluate and test model
│
├── tests/
│   ├── test_extraction.py             # Unit tests for extraction
│   ├── test_decoders.py               # Unit tests for decoders
│   ├── test_dataset.py                # Unit tests for dataset preparation (future)
│   └── fixtures/                      # Sample transactions for testing
│       ├── sample_transactions.json
│       └── sample_tx_hashes.txt
│
├── configs/
│   ├── extraction_config.yaml         # Configuration for data extraction
│   └── evaluation_config.yaml         # Evaluation settings
│
├── data/                              # Gitignored (generated locally)
│   ├── raw/                           # Raw transaction data from node
│   ├── processed/                     # Decoded structured data
│   └── datasets/                      # train/val/test splits
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
│
├── models/                            # Gitignored (generated locally)
│   ├── base/                          # Downloaded base models (cached)
│   └── fine-tuned/                    # Fine-tuned adapters
│       └── eth-intent-extractor-v1/
│           ├── adapter_model.bin
│           ├── adapter_config.json
│           └── training_logs.txt
│
└── outputs/                           # Gitignored (evaluation results)
    ├── predictions/                   # Model predictions on test set
    ├── metrics/                       # Evaluation metrics (JSON, CSV)
    └── reports/                       # Human-readable evaluation reports
```

## Integration Patterns

### MVP Usage Pattern

**For Learners (Following the Cookbook):**
```bash
# 1. Clone the repository
git clone https://github.com/your-org/eth-finetuning-cookbook.git
cd eth-finetuning-cookbook

# 2. Install dependencies using uv
uv pip install -e ".[dev]"

# 3. Follow interactive notebooks OR use scripts directly
# Option A: Interactive learning (recommended for beginners)
jupyter notebook notebooks/01-data-exploration.ipynb

# Option B: Run scripts directly (for those comfortable with CLI)
```

**For ML Engineers (Script-Based Workflow):**
```bash
# 1. Extract transaction data
python scripts/fetch_transactions.py \
  --rpc-url https://mainnet.infura.io/v3/YOUR_KEY \
  --output data/raw/transactions.json \
  --tx-hashes txs.txt

# 2. Decode transactions (outputs both decoded.csv and decoded.json)
uv run python scripts/decode_transactions.py \
  --input data/raw/transactions.json \
  --output data/processed/decoded \
  --rpc-url https://mainnet.infura.io/v3/YOUR_KEY

# 3. Prepare training dataset (use decoded.json for full data preservation)
uv run python scripts/dataset/prepare_training_data.py \
  --input data/processed/decoded.json \
  --output data/datasets/ \
  --split 0.8 0.1 0.1

# 4. Run fine-tuning (TinyLlama: ~30s, Mistral: 3-4 hours on RTX 3060)
uv run python scripts/training/train_model.py \
  --config configs/training_config.yaml \
  --dataset data/datasets/ \
  --output models/fine-tuned/eth-intent-extractor-v1

# 5. Evaluate model
python scripts/evaluation/evaluate_model.py \
  --model models/fine-tuned/eth-intent-extractor-v1 \
  --test-data data/datasets/test.jsonl \
  --output outputs/metrics/results.json

# 5. Try inference on new transactions (example)
python scripts/examples/run_inference.py \
  --model models/fine-tuned/eth-intent-extractor-v1 \
  --tx-hash 0xabcd1234...
```

### Post-MVP Extensions (Educational Topics)

**Multi-Format Outputs:**
- Tutorial on training models for different output formats (narrative, flow diagrams)
- Examples showing how to adapt prompts for various use cases

**Advanced Transaction Types:**
- Extend cookbook with flash loans, arbitrage, multi-step DeFi chapters
- Add failure diagnosis tutorial using revert data

**Production Deployment (Advanced Topics):**
- Guide on wrapping fine-tuned models in REST APIs (FastAPI example)
- Tutorial on batch processing and optimization techniques
- Example integrations with block explorers (Etherscan API)

**Model Optimization:**
- Advanced tutorial: Distillation to smaller models
- Guide on quantization to INT8 for edge deployment
- Performance optimization patterns for production use

## Success Criteria

### MVP Completion Checklist
- [ ] Data extraction pipeline successfully decodes ETH, ERC-20, Uniswap V2/V3 transactions
- [ ] Dataset preparation script outputs valid instruction-tuning format
- [ ] Fine-tuning completes in under 4 hours on RTX 3060 (12GB VRAM)
- [ ] Model achieves ≥90% accuracy on amounts, addresses, and protocols
- [ ] Generated descriptions (if applicable) achieve ≥60 Flesch Reading Ease
- [ ] All 5 notebooks are runnable end-to-end
- [ ] CLI scripts work with sample data
- [ ] README provides clear quick-start instructions
- [ ] Tests pass for core extraction and decoding logic

### Target Metrics (from Brief)
1. **Accuracy**: 90% on transaction amounts, addresses, and protocol identification
2. **Readability**: 60+ Flesch Reading Ease score for natural language outputs
3. **Training Time**: <4 hours on RTX 3060 (12GB VRAM)
4. **Hardware Accessibility**: Runs on consumer-grade 16GB VRAM GPU

## Implementation Phases

### Phase 1: Data Pipeline (Week 1-2)
- Set up Ethereum RPC connection
- Implement transaction fetcher with rate limiting
- Build decoders for ETH transfers, ERC-20, Uniswap V2/V3
- Create sample dataset (100-500 transactions)
- Write unit tests for decoders

### Phase 2: Dataset Preparation (Week 2-3)
- Define intent extraction schema
- Implement prompt templates
- Build dataset preparation script
- Create train/val/test splits
- Validate data quality

### Phase 3: Fine-Tuning Setup (Week 3-4)
- Configure QLoRA with bitsandbytes
- Implement training script with HuggingFace Trainer
- Test training loop on small dataset
- Optimize hyperparameters for 12GB VRAM constraint
- Add checkpointing and logging

### Phase 4: Evaluation & Refinement (Week 4-5)
- Implement accuracy metrics
- Add readability scoring (if generating text)
- Run full training on complete dataset
- Analyze results and iterate on data quality
- Generate evaluation reports

### Phase 5: Documentation & Packaging (Week 5-6)
- Create interactive Jupyter notebooks (5 total)
- Write comprehensive README
- Document API and CLI usage
- Add example code snippets
- Record training time and memory usage benchmarks

## Open Questions & Decisions Needed

1. **Base Model Selection**: Mistral-7B vs Llama-2-7B vs Phi-2?
   - *Recommendation*: Start with Mistral-7B-Instruct for better instruction following
   
2. **Dataset Size**: How many transactions needed for 90% accuracy?
   - *Recommendation*: Start with 1,000-5,000 transactions, iterate based on metrics
   
3. **Prompt Format**: Instruction-only or instruction + input + output?
   - *Recommendation*: Use Alpaca format (instruction + input + output) for flexibility

4. **RPC Provider**: Recommend Infura/Alchemy vs local node?
   - *Recommendation*: Document both, default to Infura/Alchemy for accessibility

5. **Model Hosting**: Should we include HuggingFace Hub upload instructions?
   - *Recommendation*: Yes, as post-training step for sharing fine-tuned models

6. **Test Data Source**: Use mainnet or testnet transactions?
   - *Recommendation*: Mainnet for realism, but include testnet instructions for learners

## Risk Mitigation

| Risk                                  | Impact | Mitigation                                                                      |
| ------------------------------------- | ------ | ------------------------------------------------------------------------------- |
| Model doesn't fit in 12GB VRAM        | High   | Use 4-bit quantization, reduce batch size, enable gradient checkpointing        |
| Training exceeds 4-hour target        | Medium | Reduce dataset size, optimize data loading, use mixed precision training        |
| Accuracy below 90%                    | High   | Increase dataset quality, tune hyperparameters, consider larger base model      |
| RPC rate limits block data extraction | Medium | Implement exponential backoff, batch requests, document local node setup        |
| Decoder fails on edge cases           | Medium | Extensive testing, graceful error handling, log failures for future improvement |
| Users lack GPU access                 | Medium | Provide Google Colab notebooks with free T4 GPU instructions                    |

---

**Document Status**: Draft v1.0  
**Last Updated**: October 3, 2025  
**Next Review**: After Phase 1 implementation
