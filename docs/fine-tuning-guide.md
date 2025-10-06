# Fine-Tuning Guide

This guide walks you through fine-tuning a large language model on Ethereum transaction data using QLoRA (Quantized Low-Rank Adaptation).

## Table of Contents

1. [Overview](#overview)
2. [Understanding QLoRA](#understanding-qlora)
3. [Preparation](#preparation)
4. [Configuration](#configuration)
5. [Training Process](#training-process)
6. [Monitoring Training](#monitoring-training)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)

## Overview

### What is Fine-Tuning?

Fine-tuning adapts a pre-trained language model to a specific task by training it on domain-specific data. Instead of training from scratch (which requires enormous computational resources), we start with a model that already understands language and teach it about Ethereum transactions.

### Why QLoRA?

**QLoRA (Quantized Low-Rank Adaptation)** enables fine-tuning large models on consumer GPUs by:

1. **4-bit Quantization**: Reduces model memory from 16-bit to 4-bit precision
2. **LoRA Adapters**: Trains small adapter layers instead of all model parameters
3. **Gradient Checkpointing**: Trades computation time for memory savings

**Result**: Fine-tune a 7B parameter model on 12GB VRAM (RTX 3060) in ~4 hours

### Training Pipeline

```
Prepared Dataset → Load Base Model → Apply QLoRA → Train Adapters → Evaluate
(train.jsonl)      (4-bit quant)     (LoRA config)  (HF Trainer)    (metrics)
```

## Understanding QLoRA

### Components

#### 1. Base Model

The foundation model with general language understanding:
- **Mistral-7B-Instruct-v0.2** (recommended): 7 billion parameters, instruction-tuned
- **Llama-2-7B-chat-hf**: Alternative with strong performance
- **Size**: ~14GB (float16) → ~3.5GB (4-bit quantized)

#### 2. Quantization (4-bit)

Reduces memory by representing weights with fewer bits:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Use 4-bit precision
    bnb_4bit_compute_dtype=torch.bfloat16, # Computation in bfloat16
    bnb_4bit_quant_type="nf4",             # NormalFloat4 quantization
    bnb_4bit_use_double_quant=True,        # Double quantization
)
```

**Trade-offs**:
- ✅ 4x memory reduction
- ✅ Minimal quality loss (<2%)
- ⚠️ Slightly slower inference

#### 3. LoRA (Low-Rank Adaptation)

Adds small trainable matrices to model layers:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                                   # Rank (size of adapters)
    lora_alpha=32,                          # Scaling factor
    lora_dropout=0.05,                      # Dropout for regularization
    target_modules=["q_proj", "v_proj"],    # Which layers to adapt
    task_type="CAUSAL_LM"                   # Task type
)
```

**Key Parameters**:
- **r (rank)**: 8-64, higher = more expressive but more memory
- **lora_alpha**: Usually 2x rank, controls adaptation strength
- **target_modules**: Which attention layers to train

**Memory savings**: Train <1% of parameters (50-100M vs 7B)

### How QLoRA Works

```
┌─────────────────────────────────────────┐
│    Base Model (Frozen, 4-bit)           │
│  ┌───────────────────────────────┐      │
│  │  Transformer Layer            │      │
│  │  ┌──────────────────┐         │      │
│  │  │ Self-Attention   │◄────────┼──────┤ LoRA Adapter
│  │  │ (frozen)         │         │      │ (trainable)
│  │  └──────────────────┘         │      │ ┌──────────┐
│  │  ┌──────────────────┐         │      │ │ A (r×d)  │
│  │  │ Feed Forward     │         │      │ │ B (d×r)  │
│  │  │ (frozen)         │         │      │ └──────────┘
│  │  └──────────────────┘         │      │
│  └───────────────────────────────┘      │
└─────────────────────────────────────────┘

Forward pass: output = frozen_layer(x) + B @ A @ x
```

Only the small A and B matrices are trained!

## Preparation

### 1. Verify Hardware

```bash
# Check GPU
nvidia-smi

# Verify CUDA in PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected: CUDA available, 12GB+ VRAM free

### 2. Prepare Dataset

Ensure you have prepared training data:

```bash
ls data/datasets/
# Should show: train.jsonl, validation.jsonl, test.jsonl
```

If not, prepare your dataset:

```bash
python scripts/dataset/prepare_training_data.py \
  --input data/processed/decoded.csv \
  --output data/datasets/ \
  --split 0.7 0.15 0.15
```

### 3. Verify Dataset Format

```bash
# Check first line of training data
head -1 data/datasets/train.jsonl | python -m json.tool
```

Expected format:
```json
{
  "instruction": "Extract the structured intent from this Ethereum transaction:",
  "input": "{...transaction data...}",
  "output": "{...structured intent...}"
}
```

### 4. Estimate Training Time

**Formula**: `time ≈ (num_samples × epochs × seq_length) / (GPU_throughput)`

**For RTX 3060 (12GB VRAM)**:
- 1,000 samples: ~30 minutes
- 5,000 samples: ~2.5 hours
- 10,000 samples: ~5 hours

**Speedup strategies**:
- Reduce `max_seq_length`: 2048 → 1024 (2x faster)
- Increase `batch_size` if you have VRAM: 1 → 2 (2x faster)
- Reduce epochs: 3 → 2

## Configuration

### Default Configuration

The default config (`configs/training_config.yaml`) is optimized for 12GB VRAM:

```yaml
# Model configuration
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"
  cache_dir: "./models/base"
  
# QLoRA configuration
lora:
  r: 16                              # Rank (adapter size)
  alpha: 32                          # LoRA alpha (scaling)
  dropout: 0.05                      # Dropout rate
  target_modules: 
    - "q_proj"                       # Query projection
    - "v_proj"                       # Value projection
  bias: "none"
  task_type: "CAUSAL_LM"
  
# 4-bit quantization
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  
# Training hyperparameters
training:
  output_dir: "models/fine-tuned/eth-intent-extractor-v1"
  num_train_epochs: 3
  per_device_train_batch_size: 1     # Micro batch
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 16    # Effective batch = 16
  learning_rate: 2.0e-4
  max_grad_norm: 0.3
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  
# Data configuration
data:
  max_seq_length: 2048
  dataset_text_field: "text"
  
# Optimization
optimization:
  gradient_checkpointing: true       # Save memory
  optim: "paged_adamw_32bit"        # Memory-efficient optimizer
  fp16: false
  bf16: true                         # Use bfloat16 if supported
  
# Logging and checkpointing
logging:
  logging_steps: 10
  save_strategy: "steps"
  save_steps: 500
  evaluation_strategy: "steps"
  eval_steps: 500
  save_total_limit: 3                # Keep last 3 checkpoints
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
```

### Customization for Different Hardware

#### For 16GB VRAM (RTX 3080/3090)

```yaml
training:
  per_device_train_batch_size: 2    # Increase batch size
  gradient_accumulation_steps: 8    # Reduce accumulation

lora:
  r: 32                               # Larger adapters
  target_modules:
    - "q_proj"
    - "k_proj"                        # Add key projection
    - "v_proj"
    - "o_proj"                        # Add output projection
```

#### For 24GB+ VRAM (RTX 4090, A5000)

```yaml
model:
  base_model: "meta-llama/Llama-2-13b-chat-hf"  # Larger model

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4

lora:
  r: 64                               # Larger rank
  target_modules: "all-linear"        # Target all linear layers

data:
  max_seq_length: 4096                # Longer sequences
```

#### For 8GB VRAM (RTX 2080, RTX 3070)

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32   # Maintain effective batch

lora:
  r: 8                                # Smaller adapters

data:
  max_seq_length: 1024                # Shorter sequences

optimization:
  gradient_checkpointing: true        # Essential!
```

### Hyperparameter Explanations

#### Learning Rate

**Default**: `2e-4` (0.0002)

- **Too high**: Training unstable, loss spikes
- **Too low**: Training slow, poor convergence
- **Rule of thumb**: Start with 2e-4, decrease if unstable

#### Batch Size

**Effective batch size** = `batch_size × gradient_accumulation_steps`

**Default**: 1 × 16 = 16

- **Larger batches**: More stable gradients, faster training
- **Smaller batches**: More noise, may help generalization
- **Memory constraint**: Increase accumulation, not batch_size

#### LoRA Rank (r)

**Default**: 16

- **r=8**: Fastest, least memory, may underfit
- **r=16**: Good balance (recommended)
- **r=32-64**: More expressive, more memory
- **r>64**: Diminishing returns

#### Number of Epochs

**Default**: 3

- **Too few**: Underfitting
- **Too many**: Overfitting
- **Monitor**: Validation loss should decrease

## Training Process

### Step 1: Start Training

Basic training command:

```bash
python scripts/training/train_model.py \
  --config configs/training_config.yaml \
  --dataset data/datasets/ \
  --output models/fine-tuned/eth-intent-extractor-v1
```

With custom parameters:

```bash
python scripts/training/train_model.py \
  --config configs/training_config.yaml \
  --dataset data/datasets/ \
  --output models/fine-tuned/my-model \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --num-epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --gradient-accumulation 16
```

Using uv (recommended):

```bash
uv run python scripts/training/train_model.py \
  --config configs/training_config.yaml \
  --dataset data/datasets/ \
  --output models/fine-tuned/eth-intent-extractor-v1
```

### Step 2: What Happens During Training

1. **Model Loading** (2-5 minutes):
   - Downloads base model (if not cached)
   - Applies 4-bit quantization
   - Initializes LoRA adapters

2. **Dataset Loading** (30 seconds):
   - Loads and tokenizes training data
   - Creates data loaders

3. **Training Loop** (2-4 hours):
   - Iterates through dataset
   - Computes loss and gradients
   - Updates LoRA parameters
   - Evaluates on validation set periodically

4. **Saving** (1-2 minutes):
   - Saves LoRA adapters (~100-200MB)
   - Saves tokenizer config
   - Writes training logs

### Step 3: Expected Output

```
Loading base model: mistralai/Mistral-7B-Instruct-v0.2
Applying 4-bit quantization...
Model loaded: 3.5GB VRAM

Loading dataset...
Train samples: 7000
Validation samples: 1500
Test samples: 1500

Initializing LoRA adapters...
Trainable parameters: 54,525,952 (0.78% of total)

Starting training...

Epoch 1/3:
  Step 10/437 | Loss: 2.1234 | LR: 1.2e-5 | Time: 25s
  Step 20/437 | Loss: 1.8923 | LR: 2.4e-5 | Time: 50s
  ...
  Step 437/437 | Loss: 0.3421 | LR: 1.9e-4 | Time: 18m
  
Validation: Loss: 0.4123 | Accuracy: 0.87

Epoch 2/3:
  ...

Training complete!
Total time: 3h 42m
Peak VRAM: 11.2GB
Final validation loss: 0.2891
Final validation accuracy: 0.92

Model saved to: models/fine-tuned/eth-intent-extractor-v1/
```

## Monitoring Training

### Real-Time Monitoring

#### 1. Watch VRAM Usage

In a separate terminal:

```bash
# Update every second
watch -n 1 nvidia-smi

# Or continuously
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1
```

**What to look for**:
- Memory usage should be <12GB (for 12GB card)
- GPU utilization should be >80%
- Temperature should be <85°C

#### 2. Monitor Training Logs

```bash
tail -f models/fine-tuned/eth-intent-extractor-v1/training_logs.txt
```

#### 3. TensorBoard (Optional)

If you enable TensorBoard logging:

```bash
tensorboard --logdir models/fine-tuned/eth-intent-extractor-v1/logs
```

Open http://localhost:6006 in your browser.

### Understanding Metrics

#### Training Loss

**Expected pattern**: Steadily decreasing

```
Epoch 1: 2.1 → 1.5 → 1.0 → 0.6 → 0.4
Epoch 2: 0.4 → 0.3 → 0.25 → 0.2
Epoch 3: 0.2 → 0.18 → 0.15 → 0.12
```

**Warning signs**:
- 📈 **Increasing loss**: Learning rate too high, reduce it
- 📊 **Plateauing early**: Learning rate too low, or model converged
- 🔥 **NaN/Inf**: Numerical instability, restart with lower LR

#### Validation Loss

**Expected**: Should track training loss but slightly higher

```
Training Loss | Validation Loss
   0.40      |     0.45        ✅ Good
   0.20      |     0.25        ✅ Good  
   0.10      |     0.50        ❌ Overfitting!
```

**If validation loss increases**:
- Stop training (early stopping)
- Reduce epochs
- Increase dropout
- Get more training data

#### Learning Rate

**Schedule**: Warmup → Constant/Cosine → Decay

```
LR
 │      Warmup    Constant/Cosine      Decay
 │      ┌────┐   ┌──────────────┐   ┌────┐
 │     /      \  /                \ /      \
 │    /        \/                  ×        \
 │___/_________________________________________
     0         100              400          437 steps
```

### Checkpoints

Training saves checkpoints every 500 steps (configurable):

```
models/fine-tuned/eth-intent-extractor-v1/
├── checkpoint-500/
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └── optimizer.pt
├── checkpoint-1000/
├── checkpoint-1500/
└── checkpoint-2000/  (latest)
```

**Resuming from checkpoint**:

```bash
python scripts/training/train_model.py \
  --config configs/training_config.yaml \
  --dataset data/datasets/ \
  --output models/fine-tuned/eth-intent-extractor-v1 \
  --resume-from-checkpoint models/fine-tuned/eth-intent-extractor-v1/checkpoint-2000
```

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions** (in order of preference):

1. **Reduce batch size** (already at 1? → skip):
   ```yaml
   training:
     per_device_train_batch_size: 1
   ```

2. **Reduce sequence length**:
   ```yaml
   data:
     max_seq_length: 1024  # Down from 2048
   ```

3. **Reduce gradient accumulation** (shorter updates):
   ```yaml
   training:
     gradient_accumulation_steps: 8  # Down from 16
   ```

4. **Reduce LoRA rank**:
   ```yaml
   lora:
     r: 8  # Down from 16
   ```

5. **Enable gradient checkpointing** (should already be on):
   ```yaml
   optimization:
     gradient_checkpointing: true
   ```

6. **Close other GPU applications**:
   ```bash
   # Kill other Python processes
   pkill -9 python
   
   # Check what's using GPU
   nvidia-smi
   ```

### Issue 2: Loss Not Decreasing

**Symptoms**: Loss stays constant or decreases very slowly

**Solutions**:

1. **Increase learning rate**:
   ```yaml
   training:
     learning_rate: 5.0e-4  # Up from 2e-4
   ```

2. **Check dataset format**:
   ```bash
   python -c "import json; data = [json.loads(l) for l in open('data/datasets/train.jsonl')]; print(data[0])"
   ```

3. **Verify LoRA targets**:
   ```python
   # Print trainable parameters
   model.print_trainable_parameters()
   # Should show ~50-100M trainable (0.5-1.5% of total)
   ```

4. **Increase LoRA rank**:
   ```yaml
   lora:
     r: 32  # Up from 16
   ```

### Issue 3: Training Too Slow

**Expected speed**: ~20-30 samples/second on RTX 3060

**Solutions**:

1. **Reduce sequence length**:
   ```yaml
   data:
     max_seq_length: 1024
   ```

2. **Increase batch size** (if VRAM available):
   ```yaml
   training:
     per_device_train_batch_size: 2
     gradient_accumulation_steps: 8
   ```

3. **Use Flash Attention** (if supported):
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       ...,
       attn_implementation="flash_attention_2"
   )
   ```

4. **Check GPU utilization**:
   ```bash
   nvidia-smi dmon
   # GPU util should be >80%
   ```

### Issue 4: Validation Loss Increasing (Overfitting)

**Symptoms**: Training loss decreases, validation loss increases

**Solutions**:

1. **Early stopping** (stop training now!)

2. **Reduce epochs**:
   ```yaml
   training:
     num_train_epochs: 2  # Down from 3
   ```

3. **Increase dropout**:
   ```yaml
   lora:
     dropout: 0.1  # Up from 0.05
   ```

4. **Get more training data** (best solution)

5. **Use data augmentation**:
   - Paraphrase instructions
   - Add noise to inputs
   - Synthetic examples

### Issue 5: Model Loading Fails

**Error**: `OSError: Unable to load weights` or `HTTP Error 401/403`

**Solutions**:

1. **Check model name**:
   ```bash
   # Verify model exists on HF Hub
   curl https://huggingface.co/api/models/mistralai/Mistral-7B-Instruct-v0.2
   ```

2. **Authenticate for gated models** (Llama-2):
   ```bash
   huggingface-cli login
   # Enter your HF token
   ```

3. **Check disk space**:
   ```bash
   df -h
   # Need ~20GB free for model cache
   ```

4. **Clear cache and retry**:
   ```bash
   rm -rf ~/.cache/huggingface/hub/
   ```

## Advanced Topics

### Custom LoRA Configurations

#### Target More Layers

For better quality (uses more VRAM):

```yaml
lora:
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

Or use `all-linear`:
```yaml
lora:
  target_modules: "all-linear"
```

#### Adaptive Rank (DoRA)

Decomposed LoRA with magnitude and direction:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,  # Enable DoRA
    target_modules=["q_proj", "v_proj"],
)
```

### Learning Rate Scheduling

#### Cosine Annealing (Default)

```yaml
training:
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03  # 3% of steps for warmup
```

#### Linear Decay

```yaml
training:
  lr_scheduler_type: "linear"
  warmup_steps: 100
```

#### Constant with Warmup

```yaml
training:
  lr_scheduler_type: "constant_with_warmup"
  warmup_steps: 100
```

### Multi-GPU Training

If you have multiple GPUs:

```bash
# Automatic detection
python scripts/training/train_model.py \
  --config configs/training_config.yaml \
  --dataset data/datasets/ \
  --output models/fine-tuned/eth-intent-extractor-v1

# Will use all available GPUs via DataParallel
```

Or use DeepSpeed:

```yaml
# configs/deepspeed_config.json
{
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 2
  }
}
```

```bash
deepspeed scripts/training/train_model.py \
  --deepspeed configs/deepspeed_config.json \
  --config configs/training_config.yaml
```

### Experiment Tracking

#### Weights & Biases

```bash
pip install wandb
wandb login
```

```python
# In training script
import wandb

wandb.init(
    project="eth-finetuning",
    name="mistral-7b-v1",
    config=config
)
```

#### MLflow

```bash
pip install mlflow
```

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("eth-intent-extraction")
```

### Hyperparameter Tuning

Use Ray Tune or Optuna for automated tuning:

```bash
python scripts/training/tune_hyperparameters.py \
  --num-trials 20 \
  --config configs/training_config.yaml
```

Searches over:
- Learning rate: [1e-5, 5e-4]
- LoRA rank: [8, 16, 32, 64]
- Batch size: [1, 2, 4]
- Dropout: [0.0, 0.05, 0.1]

## Next Steps

After training:

1. **Evaluate Your Model**: [Evaluation Guide](evaluation-guide.md)
2. **Run Inference**: `scripts/examples/run_inference.py`
3. **Analyze Results**: `notebooks/05-evaluation.ipynb`

---

**Previous**: [Data Extraction Guide](data-extraction-guide.md) | **Next**: [Evaluation Guide](evaluation-guide.md) →
