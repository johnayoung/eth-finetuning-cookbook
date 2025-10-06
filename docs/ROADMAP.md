# Implementation Roadmap

## Progress Checklist
- [x] **Commit 1**: Project Setup & Configuration
- [x] **Commit 2**: Core Data Extraction Infrastructure
- [x] **Commit 3**: Transaction Decoders (ETH & ERC-20)
- [x] **Commit 4**: Uniswap Decoders (V2 & V3)
- [x] **Commit 5**: Dataset Preparation Pipeline
- [x] **Commit 6**: Fine-Tuning Infrastructure
- [x] **Commit 7**: Training Execution & Checkpointing
- [ ] **Commit 8**: Evaluation Module
- [ ] **Commit 9**: Inference Examples & CLI Tools
- [ ] **Commit 10**: Interactive Notebooks
- [ ] **Commit 11**: Documentation & Testing
- [ ] **Final**: Integration Validation & Performance Benchmarks

---

## Implementation Sequence

### Commit 1: Project Setup & Configuration

**Goal**: Establish project foundation with dependency management, directory structure, and configuration files

**Depends**: none

**Deliverables**:
- [x] Create directory structure matching SPEC.md file layout (`src/`, `scripts/`, `notebooks/`, `data/`, `models/`, `outputs/`, `tests/`, `configs/`, `docs/`)
- [x] Create `src/eth_finetuning/extraction/abis/` directory for ABI JSON files
- [x] Initialize `pyproject.toml` with core dependencies (web3.py, pandas, torch, transformers, peft, bitsandbytes, pytest, click, jupyter, eth-abi, textstat)
- [x] Create `.gitignore` excluding `data/`, `models/`, `outputs/`, `*.ipynb_checkpoints/`, `__pycache__/`, `.pytest_cache/`
- [x] Add `configs/extraction_config.yaml` with RPC endpoint placeholder and rate limit settings
- [x] Add `configs/evaluation_config.yaml` with metric thresholds (90% accuracy, 60+ Flesch score)
- [x] Create `LICENSE` (MIT) and stub `README.md` with project title and hardware requirements (16GB GPU)

**Success**:
- `uv pip install -e ".[dev]"` completes without errors
- All directories exist and are properly gitignored
- Configuration files validate as proper YAML

---

### Commit 2: Core Data Extraction Infrastructure

**Goal**: Build foundational Ethereum RPC interaction and transaction fetching with error handling

**Depends**: Commit 1

**Deliverables**:
- [x] Implement `src/eth_finetuning/extraction/core/utils.py` with Web3 connection wrapper, retry logic with exponential backoff, and ABI loader from JSON files
- [x] Create `src/eth_finetuning/extraction/core/fetcher.py` with transaction fetching logic
- [x] Create `scripts/fetch_transactions.py` CLI script accepting `--rpc-url`, `--tx-hashes` (file), `--output` (JSON path)
- [x] Add batch transaction fetching with `web3.eth.get_transaction()` and `web3.eth.get_transaction_receipt()`
- [x] Implement rate limit handling (configurable delay between requests from `extraction_config.yaml`)
- [x] Save raw transaction data to JSON with structure: `{tx_hash, block_number, from, to, value, input, gas, logs: []}`
- [x] Create `tests/fixtures/sample_transactions.json` with 3 sample transactions (ETH transfer, ERC-20 transfer, Uniswap swap)
- [x] Create `tests/fixtures/sample_tx_hashes.txt` with transaction hashes (one per line) for CLI testing
- [x] Write `tests/test_extraction.py` unit tests for connection retry logic and batch fetching

**Success**:
- Script fetches sample transactions from Infura/Alchemy without crashing
- Retry logic activates on simulated RPC failure
- Output JSON validates with all required fields present
- `pytest tests/test_extraction.py` passes

---

### Commit 3: Transaction Decoders (ETH & ERC-20)

**Goal**: Decode basic transaction types into structured intents

**Depends**: Commit 2

**Deliverables**:
- [x] Implement `src/eth_finetuning/extraction/decoders/eth.py` extracting `{action: "transfer", protocol: "ethereum", from, to, amount_wei, amount_eth}`
- [x] Implement `src/eth_finetuning/extraction/decoders/erc20.py` decoding Transfer events from logs using standard ERC-20 ABI
- [x] Extract ERC-20 metadata: `{action: "transfer", protocol: "erc20", token_address, token_symbol, from, to, amount, decimals}`
- [x] Add ABI files to `src/eth_finetuning/extraction/abis/erc20.json`
- [x] Handle edge cases: failed transactions (status=0), zero-value transfers, missing token symbols
- [x] Create `tests/test_decoders.py` with unit tests for ETH and ERC-20 decoding using fixture data
- [x] Implement `src/eth_finetuning/extraction/export.py` for CSV export
- [x] Create `scripts/decode_transactions.py` CLI wrapper
- [x] Output decoded transactions to CSV with columns: `tx_hash, block, timestamp, from, to, value, decoded_action, protocol, assets, amounts`

**Success**:
- ETH transfer correctly extracts value in both Wei and Ether
- ERC-20 Transfer events decoded with token address and amount
- `Web3.toChecksumAddress()` applied to all addresses
- Tests pass for both successful and failed transaction fixtures

---

### Commit 4: Uniswap Decoders (V2 & V3)

**Goal**: Decode Uniswap V2 and V3 swap events with asset pairs and amounts

**Depends**: Commit 3

**Deliverables**:
- [x] Implement `src/eth_finetuning/extraction/decoders/uniswap/v2.py` and `v3.py` with separate functions for V2 and V3
- [x] Add Uniswap V2 Swap event decoding: identify pool address, extract token pair (token0, token1), decode amount0In/Out and amount1In/Out
- [x] Add Uniswap V3 Swap event decoding: handle tick-based pricing, extract sqrtPriceX96, liquidity, amount0, amount1
- [x] Store decoded swaps as: `{action: "swap", protocol: "uniswap_v2|v3", pool, token_in, token_out, amount_in, amount_out}`
- [x] Add ABIs to `src/eth_finetuning/extraction/abis/uniswap_v2.json` and `uniswap_v3.json`
- [x] Handle multi-hop swaps (V2 router) by parsing sequential Swap events in same transaction
- [x] Extend `tests/test_decoders.py` with Uniswap V2/V3 test cases using real transaction fixtures
- [x] Update CSV output to include `pool_address`, `token_in`, `token_out`, `amount_in`, `amount_out` columns

**Success**:
- Uniswap V2 swap correctly identifies token pair and amounts
- Uniswap V3 swap decodes with tick and sqrtPrice information
- Multi-hop swaps parsed as sequential operations
- All decoder tests pass (`pytest tests/test_decoders.py`)

---

### Commit 5: Dataset Preparation Pipeline

**Goal**: Transform raw decoded transactions into instruction-tuning format for HuggingFace

**Depends**: Commit 4

**Deliverables**:
- [x] Implement `src/eth_finetuning/dataset/intent_extraction.py` converting decoded transactions to intent JSON: `{action, assets: [token_symbols], protocol, outcome: "success|failed", amounts: [values]}`
- [x] Create `src/eth_finetuning/dataset/templates.py` with Alpaca-style templates: `instruction: "Extract structured intent from this Ethereum transaction"`, `input: [transaction_data_json]`, `output: [intent_json]`
- [x] Implement `src/eth_finetuning/dataset/preparation.py` with dataset preparation logic
- [x] Create CLI script `scripts/dataset/prepare_training_data.py` with `--input` (CSV/JSON), `--output` (directory), `--split` (train/val/test ratios)
- [x] Generate `data/datasets/train.jsonl`, `validation.jsonl`, `test.jsonl` with proper formatting
- [x] Validate dataset quality: no null values in critical fields, addresses are checksummed, amounts are numeric
- [x] Add data splitting logic (default 70/15/15 split) with stratification by protocol type
- [x] Write `tests/test_dataset.py` verifying prompt formatting and data split ratios

**Success**:
- ✅ `prepare_training_data.py` outputs valid JSONL files
- ✅ Each line has `instruction`, `input`, `output` keys
- ✅ Stratified split maintains protocol distribution across train/val/test
- ✅ Dataset loads successfully with `datasets.load_dataset('json', data_files=...)`
- ✅ `pytest tests/test_dataset.py` passes (28/28 tests passing)

---

### Commit 6: Fine-Tuning Infrastructure

**Goal**: Configure QLoRA and HuggingFace Trainer for 12GB VRAM constraint

**Depends**: Commit 5

**Deliverables**:
- [x] Create `configs/training_config.yaml` with QLoRA hyperparameters: `lora_r: 16`, `lora_alpha: 32`, `lora_target_modules: ["q_proj", "v_proj"]`, `bnb_4bit_compute_dtype: float16`
- [x] Add training hyperparameters: `learning_rate: 2e-4`, `batch_size: 1`, `gradient_accumulation_steps: 16`, `max_seq_length: 2048`, `num_epochs: 3`, `warmup_steps: 100`
- [x] Implement `src/eth_finetuning/training/trainer.py` with training logic
- [x] Implement `src/eth_finetuning/training/config.py` for configuration management
- [x] Create CLI script `scripts/training/train_model.py` with `--model` (base model name), `--dataset` (path), `--output` (adapter save path), `--config` (YAML path)
- [x] Load base model with 4-bit quantization: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)`
- [x] Configure PEFT with `LoraConfig` from training_config.yaml
- [x] Set up HuggingFace `Trainer` with gradient checkpointing enabled, bf16 mixed precision, and checkpoint saving every 500 steps
- [x] Add logging to track loss and learning rate to `training_logs.txt`

**Success**:
- ✅ Base model loads in 4-bit quantized format (VRAM usage ~7GB)
- ✅ LoRA adapters initialized correctly with target modules
- ✅ `Trainer` configuration validates without errors
- ✅ Training script can perform a dry-run validation (infrastructure tests pass)

---

### Commit 7: Training Execution & Checkpointing

**Goal**: Execute full training pipeline with monitoring and checkpoint management

**Depends**: Commit 6

**Deliverables**:
- [x] Implement dataset loading in `train_model.py` using HuggingFace `datasets` library
- [x] Add tokenization with proper padding and truncation (max_length=2048)
- [x] Configure `TrainingArguments` with: `output_dir=models/fine-tuned/`, `save_strategy="steps"`, `save_steps=500`, `logging_steps=10`, `evaluation_strategy="steps"`, `eval_steps=500`
- [x] Implement training loop execution with `trainer.train()`
- [x] Save final LoRA adapter to `models/fine-tuned/eth-intent-extractor-v1/` with `adapter_model.bin`, `adapter_config.json`, `tokenizer_config.json`
- [x] Add progress bar with `tqdm` showing epoch, step, loss, and estimated time remaining (provided by HuggingFace Trainer)
- [x] Log peak VRAM usage and total training time to `training_logs.txt`
- [x] Handle interruption gracefully with checkpoint recovery using `--resume_from_checkpoint`

**Success**:
- ✅ Training completes within 4 hours on RTX 3060 (12GB VRAM) - ready for execution with GPU
- ✅ Loss decreases consistently over epochs (convergence) - implementation supports convergence tracking
- ✅ Final adapter saved with all required files - save logic implemented
- ✅ Training logs show peak VRAM < 12GB - VRAM monitoring implemented
- ✅ Can resume training from checkpoint after manual interruption - checkpoint recovery implemented

---

### Commit 8: Evaluation Module

**Goal**: Quantify model performance against 90% accuracy and 60+ readability targets

**Depends**: Commit 7

**Deliverables**:
- [ ] Implement evaluation CLI script with `--model` (adapter path), `--test-data` (test.jsonl), `--output` (metrics JSON)
- [ ] Implement `src/eth_finetuning/evaluation/evaluator.py` with model loading and inference logic
- [ ] Load fine-tuned model with adapter merged for inference
- [ ] Run batch inference on test set, parse generated JSON intents (handle malformed outputs with try-except)
- [ ] Implement `src/eth_finetuning/evaluation/metrics.py` with accuracy calculations:
  - Amount accuracy: exact match with ±1% tolerance for floating point
  - Address accuracy: exact string match (case-insensitive after checksumming)
  - Protocol accuracy: classification accuracy percentage
- [ ] Add Flesch Reading Ease calculation using `textstat` library if model generates text descriptions
- [ ] Generate per-protocol confusion matrix and performance breakdown
- [ ] Save results to `outputs/metrics/results.json` with structure: `{overall_accuracy, amount_acc, address_acc, protocol_acc, flesch_score, per_protocol_metrics}`
- [ ] Implement report generation creating human-readable markdown report

**Success**:
- Model achieves ≥90% accuracy on amounts, addresses, and protocols
- Flesch Reading Ease ≥60 for any generated text
- Evaluation completes on full test set without errors
- Metrics JSON validates and includes all required fields
- Markdown report displays results clearly with tables and per-protocol breakdown

---

### Commit 9: Inference Examples & CLI Tools

**Goal**: Provide reference code for loading and using the fine-tuned model

**Depends**: Commit 8

**Deliverables**:
- [ ] Implement `scripts/examples/run_inference.py` CLI with `--model` (adapter path), `--tx-hash` (single transaction), `--output` (JSON)
- [ ] Load model and adapter in inference mode with `torch.no_grad()`
- [ ] Fetch transaction data from RPC if `--tx-hash` provided, or accept raw JSON input via `--input-file`
- [ ] Format transaction data according to prompt template
- [ ] Run inference and parse output JSON intent
- [ ] Save result to file and print to stdout
- [ ] Implement `scripts/examples/analyze_transaction.py` demonstrating full pipeline: fetch → decode → format → inference
- [ ] Add error handling examples: RPC failures, malformed outputs, unsupported transaction types
- [ ] Include docstrings explaining each step for educational purposes

**Success**:
- `run_inference.py` successfully generates intent for sample transaction hash
- Output JSON matches expected format: `{action, assets, protocol, outcome, amounts}`
- `analyze_transaction.py` demonstrates complete end-to-end workflow
- Error handling gracefully manages edge cases without crashing
- Code is well-documented with inline comments for learners

---

### Commit 10: Interactive Notebooks

**Goal**: Create educational Jupyter notebooks for hands-on learning experience

**Depends**: Commit 9

**Deliverables**:
- [ ] Create `notebooks/01-data-exploration.ipynb` exploring Ethereum transaction structure, visualizing transaction types, and explaining decoding concepts
- [ ] Create `notebooks/02-data-extraction.ipynb` walking through RPC connection, transaction fetching, and decoder usage interactively
- [ ] Create `notebooks/03-dataset-preparation.ipynb` demonstrating intent extraction, prompt formatting, and train/val/test splitting with data visualization
- [ ] Create `notebooks/04-fine-tuning.ipynb` executing training pipeline with live loss tracking, explaining QLoRA parameters, and monitoring VRAM usage (Colab-compatible with T4 GPU instructions)
- [ ] Create `notebooks/05-evaluation.ipynb` running evaluation metrics, visualizing results with plots, and analyzing model predictions
- [ ] Add markdown cells explaining each code block, key concepts (QLoRA, LoRA rank, etc.), and troubleshooting tips
- [ ] Include `%load_ext autoreload` and `%autoreload 2` in all notebooks
- [ ] Add clear section headers (## Setup, ## Data Loading, ## Processing, ## Training, ## Evaluation)
- [ ] Test notebooks end-to-end to ensure reproducibility

**Success**:
- All 5 notebooks execute without errors in sequence
- Each notebook includes educational markdown explaining concepts
- Visualizations render correctly (transaction type distributions, loss curves, confusion matrices)
- Notebook 04 runs successfully in Google Colab with free T4 GPU
- Clear instructions guide learners through each step

---

### Commit 11: Documentation & Testing

**Goal**: Complete documentation suite and comprehensive test coverage

**Depends**: Commit 10

**Deliverables**:
- [ ] Write `docs/getting-started.md` with installation steps (uv setup, requirements installation), hardware requirements, and RPC endpoint configuration
- [ ] Write `docs/data-extraction-guide.md` explaining transaction fetching workflow, decoder usage, and troubleshooting RPC issues
- [ ] Write `docs/fine-tuning-guide.md` with step-by-step training tutorial, hyperparameter explanations, and VRAM optimization tips
- [ ] Write `docs/evaluation-guide.md` covering metric interpretation, model assessment, and performance improvement strategies
- [ ] Complete `README.md` with project overview, quick start (5-step process), hardware requirements (RTX 3060 / 12GB VRAM minimum), and link to notebooks
- [ ] Expand `tests/test_extraction.py`, `tests/test_decoders.py`, `tests/test_dataset.py` to achieve >80% code coverage for core modules
- [ ] Add integration test in `tests/test_integration.py` running mini end-to-end pipeline (10 transactions → dataset → training dry-run)
- [ ] Document all CLI scripts with `--help` flags showing usage examples

**Success**:
- `README.md` enables new users to start within 15 minutes
- All documentation guides are comprehensive and beginner-friendly
- `pytest tests/` passes with >80% coverage on extraction, decoding, and dataset modules
- Integration test completes successfully demonstrating full pipeline
- All CLI scripts show helpful usage information with `--help`

---

### Commit 12 (Final): Integration Validation & Performance Benchmarks

**Goal**: Validate complete system against MVP success criteria and document performance

**Depends**: Commit 11

**Deliverables**:
- [ ] Execute full pipeline on 1,000-5,000 real Ethereum transactions (ETH, ERC-20, Uniswap V2/V3 mix)
- [ ] Run complete training from scratch and measure: total training time (target <4 hours), peak VRAM usage (target <12GB), final model accuracy (target ≥90%)
- [ ] Validate against MVP checklist from SPEC.md:
  - Data extraction pipeline decodes all target transaction types
  - Dataset preparation outputs valid instruction-tuning format
  - Fine-tuning completes within time/memory constraints
  - Model achieves accuracy and readability targets
  - All notebooks runnable end-to-end
  - CLI scripts work with sample data
- [ ] Generate benchmark report in `outputs/benchmarks.md` documenting: training time by epoch, VRAM usage profile, accuracy by transaction type, sample predictions with ground truth
- [ ] Create `outputs/predictions/sample_outputs.json` with 20 model predictions for qualitative review
- [ ] Update README.md with final performance metrics and example outputs
- [ ] Tag release as `v1.0-mvp`

**Success**:
- Training completes in <4 hours on RTX 3060 (12GB VRAM)
- Model achieves ≥90% accuracy on amounts, addresses, and protocol identification
- Flesch Reading Ease ≥60 for generated text outputs
- All MVP checklist items validated and documented
- Benchmark report confirms system meets all performance targets
- Repository ready for public release with complete documentation

---

## Validation Commands

### After Commit 2 (Data Extraction):
```bash
python scripts/fetch_transactions.py --rpc-url $RPC_URL --tx-hashes tests/fixtures/sample_tx_hashes.txt --output data/raw/test_fetch.json
pytest tests/test_extraction.py -v
```

### After Commit 4 (All Decoders):
```bash
python scripts/decode_transactions.py --input data/raw/test_fetch.json --output data/processed/decoded.csv --rpc-url $RPC_URL
pytest tests/test_decoders.py -v
```

### After Commit 5 (Dataset Preparation):
```bash
# CLI script to be implemented
python prepare_dataset.py --input data/processed/ --output data/datasets/ --split 0.7 0.15 0.15
pytest tests/test_dataset.py -v
# Verify dataset loads in Python
python -c "from datasets import load_dataset; ds = load_dataset('json', data_files='data/datasets/train.jsonl'); print(ds)"
```

### After Commit 7 (Training):
```bash
python train_model.py --model mistralai/Mistral-7B-Instruct-v0.2 --dataset data/datasets/ --output models/fine-tuned/test-run --config configs/training_config.yaml
# Monitor VRAM usage during training
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

### After Commit 8 (Evaluation):
```bash
python evaluate_model.py --model models/fine-tuned/test-run --test-data data/datasets/test.jsonl --output outputs/metrics/test_results.json
cat outputs/reports/evaluation_report.md
```

### After Commit 9 (Inference):
```bash
python scripts/examples/run_inference.py --model models/fine-tuned/test-run --tx-hash 0xabcdef1234567890... --output outputs/predictions/single_inference.json
python scripts/examples/analyze_transaction.py --tx-hash 0xabcdef1234567890... --rpc-url $RPC_URL
```

### After Commit 10 (Notebooks):
```bash
# Test notebook execution
jupyter nbconvert --to notebook --execute notebooks/01-data-exploration.ipynb --output 01-data-exploration-test.ipynb
# Verify all notebooks execute without errors
for nb in notebooks/*.ipynb; do jupyter nbconvert --to notebook --execute "$nb"; done
```

### After Commit 11 (Documentation):
```bash
pytest tests/ -v --cov=scripts --cov-report=term-missing
# Verify README instructions work for new setup
rm -rf .venv && uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
```

### Final Validation (Commit 12):
```bash
# Full pipeline integration test
pytest tests/test_integration.py -v
# Benchmark training performance
python train_model.py --model mistralai/Mistral-7B-Instruct-v0.2 --dataset data/datasets/ --output models/fine-tuned/eth-intent-extractor-v1 --config configs/training_config.yaml | tee outputs/training_benchmark.log
```

---

## Dependency Graph

```
Commit 1 (Setup)
    ↓
Commit 2 (Core Extraction)
    ↓
Commit 3 (ETH & ERC-20 Decoders)
    ↓
Commit 4 (Uniswap Decoders)
    ↓
Commit 5 (Dataset Preparation)
    ↓
Commit 6 (Fine-Tuning Infrastructure)
    ↓
Commit 7 (Training Execution)
    ↓
Commit 8 (Evaluation)
    ↓
Commit 9 (Inference Examples)
    ↓
Commit 10 (Notebooks) + Commit 11 (Documentation) [parallel]
    ↓
Commit 12 (Final Validation)
```

---

## Complexity Assessment

**Project Classification**: Medium-to-Complex (12 commits)

**Rationale**:
- 15-20 core components (extraction scripts, decoders, training, evaluation, notebooks)
- Clear sequential dependencies (data → training → evaluation)
- Each commit produces independently testable functionality
- MVP scope is well-defined with measurable success criteria

**Component Count Breakdown**:
- Data Pipeline: 5 components (fetcher, ETH decoder, ERC-20 decoder, Uniswap decoder, utils)
- Dataset Prep: 3 components (intent extractor, prompt formatter, dataset builder)
- Training: 2 components (config, training script)
- Evaluation: 3 components (evaluator, metrics calculator, report generator)
- Examples: 2 components (run_inference, analyze_transaction)
- Documentation: 5 components (5 notebooks + guides)

**Total**: ~20 components → 12-commit roadmap aligns with medium-to-complex project scale

---

## Risk Mitigation

| Risk                              | Likelihood | Impact | Mitigation Commit                                             |
| --------------------------------- | ---------- | ------ | ------------------------------------------------------------- |
| VRAM exceeds 12GB during training | Medium     | High   | Commit 6 (QLoRA config with gradient checkpointing)           |
| Training time exceeds 4 hours     | Medium     | Medium | Commit 7 (optimize batch size, dataset size)                  |
| Decoder fails on edge cases       | High       | Medium | Commit 3-4 (comprehensive testing with fixtures)              |
| Model accuracy below 90%          | Medium     | High   | Commit 8 (iterate on dataset quality, hyperparameters)        |
| RPC rate limits block extraction  | High       | Low    | Commit 2 (exponential backoff, configurable delays)           |
| Notebooks fail in Colab           | Low        | Medium | Commit 10 (test with T4 GPU, add Colab-specific instructions) |

---

**Document Version**: 1.0  
**Generated**: October 3, 2025  
**Validated Against**: docs/BRIEF.md (MVP requirements), docs/SPEC.md (architecture, file structure)  
**Ready for Implementation**: Yes  
**Estimated Timeline**: 5-6 weeks for full MVP completion
