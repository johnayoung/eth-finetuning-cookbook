# Evaluation Guide

This guide explains how to evaluate your fine-tuned model's performance and interpret the results.

## Table of Contents

1. [Overview](#overview)
2. [Running Evaluation](#running-evaluation)
3. [Understanding Metrics](#understanding-metrics)
4. [Interpreting Results](#interpreting-results)
5. [Performance Analysis](#performance-analysis)
6. [Improvement Strategies](#improvement-strategies)
7. [Advanced Evaluation](#advanced-evaluation)

## Overview

### Why Evaluate?

Evaluation quantifies model performance to:
- Validate fine-tuning success
- Compare model versions
- Identify weaknesses
- Guide improvements
- Demonstrate capabilities

### Evaluation Metrics

The cookbook uses these metrics (from `configs/evaluation_config.yaml`):

1. **Overall Accuracy**: Percentage of correct predictions (target: ≥90%)
2. **Amount Accuracy**: Correctness of numerical values (±1% tolerance)
3. **Address Accuracy**: Exact address match (checksummed)
4. **Protocol Accuracy**: Protocol classification accuracy
5. **Flesch Reading Ease**: Readability score (target: ≥60)

### Evaluation Pipeline

```
Test Dataset → Load Model → Generate Predictions → Calculate Metrics → Generate Report
(test.jsonl)   (+ adapter)   (batch inference)    (accuracy, etc)     (markdown/JSON)
```

## Running Evaluation

### Basic Evaluation

```bash
python scripts/evaluation/evaluate_model.py \
  --model models/fine-tuned/eth-intent-extractor-v1 \
  --test-data data/datasets/test.jsonl \
  --output outputs/metrics/results.json
```

### With Custom Configuration

```bash
python scripts/evaluation/evaluate_model.py \
  --model models/fine-tuned/eth-intent-extractor-v1 \
  --test-data data/datasets/test.jsonl \
  --output outputs/metrics/results.json \
  --config configs/evaluation_config.yaml \
  --batch-size 4 \
  --max-samples 500
```

### Using uv

```bash
uv run python scripts/evaluation/evaluate_model.py \
  --model models/fine-tuned/eth-intent-extractor-v1 \
  --test-data data/datasets/test.jsonl \
  --output outputs/metrics/results.json
```

### Parameters

- `--model`: Path to fine-tuned model (with LoRA adapters)
- `--test-data`: Test dataset (JSONL format)
- `--output`: Where to save metrics (JSON)
- `--config`: Evaluation configuration file (optional)
- `--batch-size`: Inference batch size (default: 1)
- `--max-samples`: Limit test samples for quick evaluation
- `--generate-report`: Generate markdown report (default: true)

### Expected Output

```
Loading model: models/fine-tuned/eth-intent-extractor-v1
Model loaded successfully (3.8GB)

Loading test dataset: data/datasets/test.jsonl
Loaded 1,500 test samples

Running inference...
Progress: 100%|████████████████████| 1500/1500 [12:34<00:00, 1.99it/s]

Calculating metrics...

=== Overall Results ===
Overall Accuracy: 92.3%
Amount Accuracy: 94.1%
Address Accuracy: 98.7%
Protocol Accuracy: 91.2%
Flesch Reading Ease: 64.2

=== Per-Protocol Results ===
Ethereum:    95.2% (476/500)
ERC-20:      91.8% (459/500)
Uniswap V2:  89.3% (223/250)
Uniswap V3:  92.0% (230/250)

=== Performance Status ===
✅ Overall Accuracy: 92.3% (target: 90%)
✅ Amount Accuracy: 94.1% (target: 90%)
✅ Address Accuracy: 98.7% (target: 90%)
✅ Protocol Accuracy: 91.2% (target: 90%)
✅ Flesch Score: 64.2 (target: 60)

All targets met! 🎉

Results saved to: outputs/metrics/results.json
Report saved to: outputs/reports/evaluation_report.md
```

## Understanding Metrics

### Overall Accuracy

**Definition**: Percentage of predictions where ALL fields are correct.

**Calculation**:
```python
correct = 0
for prediction, ground_truth in zip(predictions, ground_truths):
    if prediction == ground_truth:  # Exact match
        correct += 1

overall_accuracy = correct / total * 100
```

**Target**: ≥90%

**Interpretation**:
- **≥95%**: Excellent, production-ready
- **90-95%**: Good, meets target
- **85-90%**: Acceptable, consider improvements
- **<85%**: Needs more training or better data

### Amount Accuracy

**Definition**: Percentage of correct numerical amounts with tolerance.

**Calculation**:
```python
def amounts_match(pred_amount, true_amount, tolerance=0.01):
    """Check if amounts match within ±1% tolerance."""
    return abs(pred_amount - true_amount) / true_amount <= tolerance

amount_accuracy = sum(amounts_match(p, t) for p, t in zip(preds, truths)) / total * 100
```

**Why tolerance?**: Handles floating-point rounding and slight formatting differences.

**Target**: ≥90%

**Common errors**:
- Decimal place confusion (1.0 vs 1000.0)
- Wei vs Ether conversion (1e18)
- Token decimals (USDC has 6, not 18)

### Address Accuracy

**Definition**: Percentage of correctly extracted Ethereum addresses.

**Calculation**:
```python
from web3 import Web3

def addresses_match(pred_addr, true_addr):
    """Check if addresses match (case-insensitive checksumming)."""
    try:
        pred_checksummed = Web3.to_checksum_address(pred_addr)
        true_checksummed = Web3.to_checksum_address(true_addr)
        return pred_checksummed == true_checksummed
    except:
        return False

address_accuracy = sum(addresses_match(p, t) for p, t in zip(preds, truths)) / total * 100
```

**Target**: ≥90% (often achieves 95-99%)

**Common errors**:
- Truncated addresses (0x123...456)
- Missing 0x prefix
- Wrong address extracted (sender vs receiver)

### Protocol Accuracy

**Definition**: Percentage of correct protocol classifications.

**Calculation**:
```python
protocol_accuracy = sum(
    pred["protocol"] == true["protocol"] 
    for pred, true in zip(predictions, ground_truths)
) / total * 100
```

**Protocols**:
- `ethereum`: Native ETH transfers
- `erc20`: ERC-20 token transfers
- `uniswap_v2`: Uniswap V2 swaps
- `uniswap_v3`: Uniswap V3 swaps

**Target**: ≥90%

**Confusion matrix** shows common misclassifications.

### Flesch Reading Ease

**Definition**: Readability score for generated text (if model produces descriptions).

**Formula**:
```
Flesch = 206.835 - 1.015 × (total_words / total_sentences) 
                 - 84.6 × (total_syllables / total_words)
```

**Scale**:
- **90-100**: Very easy (5th grade)
- **60-70**: Standard (8th-9th grade)  ← Target
- **30-50**: Difficult (college)
- **0-30**: Very difficult (graduate)

**Target**: ≥60 (easily understandable)

**Example outputs**:

**Good (72)**:
> "This transaction swaps 1,000 USDC for 0.5 WETH on Uniswap V2."

**Poor (38)**:
> "This transaction facilitates the decentralized exchange of 1,000 USDC tokens for approximately 0.5 WETH tokens utilizing the Uniswap V2 automated market-making protocol."

## Interpreting Results

### Results JSON Structure

```json
{
  "overall_accuracy": 0.923,
  "amount_accuracy": 0.941,
  "address_accuracy": 0.987,
  "protocol_accuracy": 0.912,
  "flesch_reading_ease": 64.2,
  
  "per_protocol": {
    "ethereum": {
      "accuracy": 0.952,
      "total": 500,
      "correct": 476
    },
    "erc20": {
      "accuracy": 0.918,
      "total": 500,
      "correct": 459
    },
    "uniswap_v2": {
      "accuracy": 0.893,
      "total": 250,
      "correct": 223
    },
    "uniswap_v3": {
      "accuracy": 0.920,
      "total": 250,
      "correct": 230
    }
  },
  
  "confusion_matrix": {
    "ethereum": {"ethereum": 476, "erc20": 24, "uniswap_v2": 0, "uniswap_v3": 0},
    "erc20": {"ethereum": 21, "erc20": 459, "uniswap_v2": 15, "uniswap_v3": 5},
    "uniswap_v2": {"ethereum": 0, "erc20": 12, "uniswap_v2": 223, "uniswap_v3": 15},
    "uniswap_v3": {"ethereum": 0, "erc20": 5, "uniswap_v2": 15, "uniswap_v3": 230}
  },
  
  "error_analysis": {
    "total_errors": 115,
    "error_breakdown": {
      "amount_error": 42,
      "address_error": 18,
      "protocol_error": 38,
      "format_error": 17
    }
  },
  
  "inference_stats": {
    "total_samples": 1500,
    "total_time_seconds": 754,
    "avg_time_per_sample": 0.503,
    "throughput_samples_per_second": 1.99
  }
}
```

### Confusion Matrix Interpretation

The confusion matrix shows misclassifications:

```
           Predicted
          ETH  ERC20  V2   V3
Actual ETH  476   24    0    0
      ERC20  21  459   15    5
       V2     0   12  223   15
       V3     0    5   15  230
```

**Insights**:
- **ETH → ERC20**: 24 errors (model confuses basic transfers)
- **ERC20 → V2**: 15 errors (misses swap events, sees only transfers)
- **V2 ↔ V3**: 30 errors total (confuses Uniswap versions)

**Action**: Add more distinguishing features in training data.

### Sample Predictions

The evaluation saves sample predictions for manual review:

```json
{
  "sample_id": 42,
  "input": {
    "tx_hash": "0x123...",
    "from": "0xABC...",
    "to": "0xDEF..."
  },
  "ground_truth": {
    "action": "swap",
    "protocol": "uniswap_v2",
    "token_in": "USDC",
    "token_out": "WETH",
    "amount_in": "1000",
    "amount_out": "0.5"
  },
  "prediction": {
    "action": "swap",
    "protocol": "uniswap_v3",  ← ERROR
    "token_in": "USDC",
    "token_out": "WETH",
    "amount_in": "1000.0",
    "amount_out": "0.5"
  },
  "correct": false,
  "error_type": "protocol_mismatch"
}
```

Saved to: `outputs/predictions/sample_outputs.json`

## Performance Analysis

### Good Performance Indicators

✅ **High accuracy across all protocols** (>90% each)
✅ **Low variance between train and validation** (not overfitting)
✅ **Fast inference** (<1 second per sample)
✅ **Confusion matrix shows diagonal dominance** (few misclassifications)
✅ **Error types are diverse** (not systematic)

### Warning Signs

⚠️ **Low protocol-specific accuracy** → Need more training data for that protocol
⚠️ **High inference time** → Model inefficiency or hardware issue
⚠️ **Systematic errors** → Training data bias or model architecture issue
⚠️ **Validation much worse than training** → Overfitting

### Benchmark Comparison

**Expected performance on RTX 3060**:

| Metric            | Target | Typical           | Excellent    |
| ----------------- | ------ | ----------------- | ------------ |
| Overall Accuracy  | 90%    | 91-93%            | >95%         |
| Amount Accuracy   | 90%    | 93-95%            | >97%         |
| Address Accuracy  | 90%    | 96-99%            | >99%         |
| Protocol Accuracy | 90%    | 89-93%            | >94%         |
| Flesch Score      | 60     | 62-68             | >70          |
| Inference Speed   | -      | 1.5-2.5 samples/s | >3 samples/s |

### Error Distribution

Analyze where errors occur:

```python
# From evaluation output
error_breakdown = {
    "amount_error": 42,      # 36% of errors
    "address_error": 18,     # 16% of errors
    "protocol_error": 38,    # 33% of errors
    "format_error": 17       # 15% of errors
}
```

**Focus improvement efforts** on the highest-error category.

## Improvement Strategies

### If Overall Accuracy < 90%

1. **Check Training Data Quality**:
   ```bash
   # Verify data format
   python scripts/dataset/validate_dataset.py data/datasets/
   ```

2. **Increase Training Duration**:
   ```yaml
   # configs/training_config.yaml
   training:
     num_train_epochs: 5  # Up from 3
   ```

3. **Improve Dataset**:
   - Add more diverse examples
   - Fix labeling errors
   - Balance protocol distribution

4. **Adjust Hyperparameters**:
   ```yaml
   lora:
     r: 32  # Increase rank
   training:
     learning_rate: 5.0e-4  # Increase LR
   ```

### If Amount Accuracy Low

**Problem**: Model struggles with numerical values.

**Solutions**:

1. **Normalize amounts in training**:
   ```python
   # Always use same decimal format
   amount = f"{value:.6f}"  # 6 decimal places
   ```

2. **Add more amount examples**:
   - Include edge cases (very large, very small)
   - Various decimal precisions

3. **Explicit amount formatting in prompts**:
   ```json
   {
     "instruction": "Extract amounts with 6 decimal precision...",
     ...
   }
   ```

### If Address Accuracy Low

**Problem**: Address extraction or formatting issues.

**Solutions**:

1. **Verify checksumming in training data**:
   ```bash
   python scripts/dataset/validate_addresses.py data/datasets/
   ```

2. **Add address validation to training**:
   ```python
   from web3 import Web3
   address = Web3.to_checksum_address(raw_address)
   ```

3. **Include address examples in prompt**:
   ```
   "Addresses must be 42 characters starting with 0x..."
   ```

### If Protocol Accuracy Low

**Problem**: Model confuses transaction types.

**Solutions**:

1. **Analyze confusion matrix**:
   ```python
   # Which protocols are confused?
   # E.g., V2 ↔ V3, ETH ↔ ERC20
   ```

2. **Add distinguishing features**:
   - For V2 vs V3: Include pool version explicitly
   - For ETH vs ERC20: Mention "native ETH" vs "token"

3. **Balance training data**:
   ```bash
   # Check distribution
   python -c "
   import json
   from collections import Counter
   data = [json.loads(l) for l in open('data/datasets/train.jsonl')]
   protocols = [json.loads(d['output'])['protocol'] for d in data]
   print(Counter(protocols))
   "
   ```

### If Flesch Score Low

**Problem**: Generated text too complex.

**Solutions**:

1. **Simplify training outputs**:
   - Use shorter sentences
   - Avoid technical jargon
   - Active voice

2. **Add readability to training objective** (advanced):
   ```python
   # Reward model for simpler language
   ```

3. **Post-process outputs**:
   ```python
   import textstat
   
   def simplify_if_needed(text):
       if textstat.flesch_reading_ease(text) < 60:
           # Simplify logic
           pass
       return text
   ```

### If Inference Too Slow

**Problem**: <1 sample/second.

**Solutions**:

1. **Increase batch size**:
   ```bash
   python scripts/evaluation/evaluate_model.py \
     --batch-size 8  # Up from 1
   ```

2. **Use Flash Attention**:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       ...,
       attn_implementation="flash_attention_2"
   )
   ```

3. **Reduce sequence length**:
   ```yaml
   data:
     max_seq_length: 1024  # Down from 2048
   ```

4. **Check GPU utilization**:
   ```bash
   nvidia-smi dmon
   # Should be >80%
   ```

## Advanced Evaluation

### A/B Testing Models

Compare two model versions:

```bash
# Evaluate Model A
python scripts/evaluation/evaluate_model.py \
  --model models/fine-tuned/model-v1 \
  --test-data data/datasets/test.jsonl \
  --output outputs/metrics/model_v1.json

# Evaluate Model B
python scripts/evaluation/evaluate_model.py \
  --model models/fine-tuned/model-v2 \
  --test-data data/datasets/test.jsonl \
  --output outputs/metrics/model_v2.json

# Compare
python scripts/evaluation/compare_models.py \
  --model-a outputs/metrics/model_v1.json \
  --model-b outputs/metrics/model_v2.json \
  --output outputs/reports/comparison.md
```

### Statistical Significance

Test if improvement is significant:

```python
from scipy import stats

# Accuracy samples from 5 runs
model_v1 = [0.92, 0.91, 0.93, 0.92, 0.91]
model_v2 = [0.94, 0.95, 0.94, 0.93, 0.95]

t_stat, p_value = stats.ttest_ind(model_v1, model_v2)
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Improvement is statistically significant!")
else:
    print("Difference may be due to chance.")
```

### Cross-Validation

Evaluate on multiple test splits:

```bash
# Create 5-fold CV splits
python scripts/dataset/create_cv_splits.py \
  --input data/datasets/combined.jsonl \
  --output data/datasets/cv/ \
  --folds 5

# Evaluate on each fold
for fold in {0..4}; do
  python scripts/evaluation/evaluate_model.py \
    --model models/fine-tuned/eth-intent-extractor-v1 \
    --test-data data/datasets/cv/test_fold_${fold}.jsonl \
    --output outputs/metrics/fold_${fold}.json
done

# Aggregate results
python scripts/evaluation/aggregate_cv_results.py \
  --results outputs/metrics/fold_*.json \
  --output outputs/metrics/cv_aggregated.json
```

### Qualitative Analysis

Manual review of predictions:

```bash
# Generate HTML report with samples
python scripts/evaluation/generate_qualitative_report.py \
  --predictions outputs/predictions/sample_outputs.json \
  --output outputs/reports/qualitative_analysis.html

# Open in browser
open outputs/reports/qualitative_analysis.html
```

### Adversarial Testing

Test on challenging cases:

```python
# Create adversarial examples
adversarial_cases = [
    # Edge case: Very small amount
    {"value": "1", "expected_eth": "0.000000000000000001"},
    
    # Edge case: Very large amount
    {"value": "999999999999999999999999", "expected_eth": "999999.999999999999999999"},
    
    # Confusion: Multi-hop swap
    {"protocol": "uniswap_v2", "hops": 3},
    
    # Rare protocol
    {"protocol": "curve", "action": "swap"}
]
```

### Continuous Evaluation

Set up automated evaluation:

```bash
# cron job: Daily evaluation
0 2 * * * cd /path/to/project && uv run python scripts/evaluation/evaluate_model.py --model models/fine-tuned/latest --test-data data/datasets/test.jsonl --output outputs/metrics/daily_$(date +\%Y\%m\%d).json
```

Track metrics over time:

```python
import matplotlib.pyplot as plt
import glob
import json

# Load all daily results
results = []
for file in sorted(glob.glob("outputs/metrics/daily_*.json")):
    with open(file) as f:
        results.append(json.load(f))

# Plot accuracy over time
accuracies = [r["overall_accuracy"] for r in results]
plt.plot(accuracies)
plt.axhline(y=0.90, color='r', linestyle='--', label='Target')
plt.title("Model Accuracy Over Time")
plt.xlabel("Days")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/reports/accuracy_trend.png")
```

## Next Steps

After evaluation:

1. **If targets met**: Deploy model or move to production testing
2. **If targets not met**: Iterate with improvement strategies above
3. **Document results**: Update README with final metrics
4. **Share findings**: Create a blog post or paper

### Production Checklist

Before deploying:

- [ ] Overall accuracy ≥90%
- [ ] All protocol accuracies ≥85%
- [ ] Inference speed acceptable (<2s per transaction)
- [ ] Tested on real-world transactions (not just test set)
- [ ] Error handling implemented
- [ ] Monitoring and logging set up
- [ ] Fallback strategy for model failures

---

**Previous**: [Fine-Tuning Guide](fine-tuning-guide.md) | **Back to**: [Getting Started](getting-started.md)
