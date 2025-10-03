# GitHub Copilot Instructions

## Development Standards

### Python Best Practices
- Use Python 3.10+ features (type hints with `|`, match statements, structural pattern matching). Always include type annotations for function signatures.
- Follow PEP 8 for code style; use `snake_case` for functions/variables, `PascalCase` for classes. Avoid mutable default arguments.
- Use context managers (`with` statements) for file I/O and resource management. Prefer pathlib over os.path for file operations.
- Handle exceptions explicitly; avoid bare `except:` clauses. Use specific exception types and provide meaningful error messages.

### web3.py Patterns
- Always use `Web3.toChecksumAddress()` before comparing or storing Ethereum addresses to prevent case-sensitivity issues.
- Implement exponential backoff with retries for RPC calls to handle rate limits and transient network failures gracefully.
- Use batch requests (`web3.eth.get_block()` with `full_transactions=True`) to minimize RPC calls and improve performance.
- Validate contract ABIs before decoding; catch `eth_abi.exceptions.DecodingError` and log problematic transaction hashes for debugging.

### pandas Guidelines
- Use vectorized operations instead of `iterrows()` or `apply()` with lambdas for better performance on large datasets.
- Explicitly set `dtype` when loading CSVs to prevent type inference issues. Use `pd.read_parquet()` for large datasets (faster than CSV).
- Chain operations with method chaining (`.pipe()`, `.assign()`) for readability. Avoid in-place operations in data pipelines.
- Always validate data quality: check for nulls (`df.isnull().sum()`), duplicates, and expected value ranges before processing.

### PyTorch & HuggingFace Patterns
- Use `torch.no_grad()` context during inference to save memory. Always call `model.eval()` before evaluation/inference.
- Implement gradient accumulation when batch size is limited by VRAM: update weights every N steps, not every batch.
- Use `Trainer` API from Transformers instead of manual training loops for automatic mixed precision, checkpointing, and logging.
- Load models with `device_map="auto"` for multi-GPU/CPU offloading. Enable gradient checkpointing for large models: `model.gradient_checkpointing_enable()`.
- Always specify `torch_dtype=torch.float16` or `torch.bfloat16` when loading models to reduce memory usage by 50%.

### QLoRA/PEFT Best Practices
- Set `bnb_4bit_compute_dtype=torch.float16` in quantization config for optimal performance vs accuracy trade-off.
- Use `lora_target_modules=["q_proj", "v_proj"]` at minimum; add `"k_proj", "o_proj"` for better results if VRAM permits.
- Set `lora_r` (rank) between 8-64: lower values save memory but may underfit; start with 16 for most tasks.
- Save only LoRA adapters (`model.save_pretrained()`) not the full model; merge adapters post-training only if deploying standalone.

### Jupyter Notebook Guidelines
- Structure notebooks with clear markdown headers (##) separating: Setup, Data Loading, Processing, Training, Evaluation.
- Include `%load_ext autoreload` and `%autoreload 2` at the top to auto-reload modules during development.
- Clear outputs before committing notebooks to git: `jupyter nbconvert --clear-output --inplace *.ipynb`.
- Use `tqdm.notebook.tqdm` instead of `tqdm.tqdm` for progress bars that render properly in notebook environments.

### Code Quality Standards
- **Syntax/Parsing Prevention**: Use `pyright` or `mypy` for static type checking. Run before committing: `pyright scripts/` or `mypy --strict scripts/`.
- **Error Handling**: Wrap RPC calls in try-except with specific exceptions (`requests.exceptions.RequestException`, `Web3Exception`). Log full stack traces with `logging.exception()`.
- **Testing**: Write unit tests for decoders with fixtures from `tests/fixtures/`. Use `pytest.mark.parametrize` for testing multiple transaction types.
- **Logging**: Use Python `logging` module, not `print()`. Set level to INFO for pipelines, DEBUG for development: `logging.basicConfig(level=logging.INFO)`.
- **GPU Memory**: Always free CUDA memory after operations: `torch.cuda.empty_cache()`. Monitor usage with `torch.cuda.memory_allocated()`.

### Project Conventions
- **File Naming**: Use descriptive names with underscores: `decode_uniswap_v3.py`, not `decoder.py`. Scripts should be verbs: `fetch_transactions.py`.
- **Imports**: Group imports: stdlib, third-party, local modules. Use absolute imports from project root, not relative imports.
- **Configuration**: Store hyperparameters in YAML files under `configs/`, not hardcoded. Load with `yaml.safe_load()`.
- **Data Paths**: Use `pathlib.Path` and construct paths relative to project root. Define `PROJECT_ROOT = Path(__file__).parent.parent` in scripts.
- **Git**: Never commit large files (models, datasets) to git. Use `.gitignore` for `data/`, `models/`, `outputs/`, `*.ipynb_checkpoints/`.
- **Documentation**: Every script must have a docstring explaining purpose, inputs, outputs, and example usage. Use Google-style docstrings.

### Ethereum-Specific Conventions
- **Address Format**: Always store addresses as checksummed strings (`0xAbC...`), never lowercase. Validate with `Web3.is_address()` before processing.
- **Wei vs Ether**: Store amounts in Wei (int) in raw data; convert to Ether (float) only for display using `Web3.from_wei(amount, 'ether')`.
- **Transaction Hashes**: Store as hex strings with `0x` prefix. Use `.hex()` method when converting from HexBytes: `tx_hash.hex()`.
- **Block Numbers**: Use integers for block numbers, not strings. Cache block timestamp lookups to avoid repeated RPC calls.
- **ABI Loading**: Store common ABIs (ERC-20, Uniswap) in `scripts/extraction/abis/` as JSON. Load once at module level, not per function call.
