"""
Model evaluation and batch inference for Ethereum intent extraction.

This module provides functionality for loading fine-tuned models with adapters,
running batch inference on test sets, and generating comprehensive evaluation
reports with accuracy metrics and confusion matrices.

Usage:
    from eth_finetuning.evaluation import evaluate_model

    results = evaluate_model(
        model_path="models/fine-tuned/eth-intent-extractor",
        test_data_path="data/datasets/test.jsonl",
        output_path="outputs/metrics/results.json"
    )
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .metrics import calculate_accuracy_metrics, calculate_per_protocol_metrics

logger = logging.getLogger(__name__)


def load_model_for_evaluation(
    model_path: str | Path,
    device_map: str = "auto",
    load_in_4bit: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load fine-tuned model with adapter for evaluation.

    Loads base model and applies LoRA adapter for inference. Model is set
    to evaluation mode with gradients disabled.

    Args:
        model_path: Path to model directory containing adapter files
        device_map: Device mapping strategy (default: "auto")
        load_in_4bit: Whether to load in 4-bit quantized format (default: True)

    Returns:
        Tuple of (model, tokenizer) ready for inference

    Raises:
        FileNotFoundError: If model path doesn't exist
        RuntimeError: If model loading fails

    Notes:
        - Model is automatically set to eval() mode
        - Uses torch.no_grad() context for inference
        - 4-bit loading reduces VRAM from ~7GB to ~4GB
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logger.info(f"Loading model from {model_path}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            # Check if this is a PEFT model (has adapter)
            adapter_config_path = model_path / "adapter_config.json"
            if adapter_config_path.exists():
                # Load base model from adapter config
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path")

                if base_model_name:
                    logger.info(f"Loading base model: {base_model_name}")
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        quantization_config=bnb_config,
                        device_map=device_map,
                        trust_remote_code=False,
                    )

                    # Load adapter
                    logger.info(f"Loading adapter from {model_path}")
                    model = PeftModel.from_pretrained(model, str(model_path))
                else:
                    raise ValueError(
                        "base_model_name_or_path not found in adapter config"
                    )
            else:
                # No adapter, load directly
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=bnb_config,
                    device_map=device_map,
                    trust_remote_code=False,
                )
        else:
            # Load without quantization
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map=device_map,
                trust_remote_code=False,
            )

        # Set to evaluation mode
        model.eval()

        logger.info("Model loaded successfully")
        return model, tokenizer  # type: ignore

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Run inference on a single prompt.

    Args:
        model: Loaded model (in eval mode)
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate (default: 512)
        temperature: Sampling temperature (default: 0.1 for deterministic)
        top_p: Nucleus sampling parameter (default: 0.9)

    Returns:
        Generated text (decoded from tokens)

    Notes:
        - Lower temperature (0.1) produces more deterministic outputs
        - Uses torch.no_grad() for memory efficiency
        - Automatically moves inputs to model device
    """
    with torch.no_grad():
        # Tokenize input
        inputs = tokenizer(  # type: ignore
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # type: ignore

        # Generate
        outputs = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            eos_token_id=tokenizer.eos_token_id,  # type: ignore
        )

        # Decode (skip input tokens)
        generated_text = tokenizer.decode(  # type: ignore
            outputs[0][inputs["input_ids"].shape[1] :],  # type: ignore
            skip_special_tokens=True,
        )

        return generated_text


def parse_json_output(text: str) -> dict[str, Any] | None:
    """
    Parse JSON output from model generation.

    Attempts to extract and parse JSON from model output, handling
    common formatting issues and malformed outputs.

    Args:
        text: Generated text that should contain JSON

    Returns:
        Parsed JSON dictionary, or None if parsing fails

    Notes:
        - Tries to find JSON object in text (looks for {...})
        - Handles extra whitespace and newlines
        - Returns None for malformed JSON (logs warning)
    """
    if not text or not isinstance(text, str):
        return None

    # Try to find JSON in the text
    text = text.strip()

    # Look for JSON object markers
    start_idx = text.find("{")
    end_idx = text.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx : end_idx + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.debug(f"Malformed JSON: {json_str[:200]}")
            return None

    # No JSON object found
    logger.warning("No JSON object found in generated text")
    return None


def batch_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_dataset: Any,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    batch_size: int = 1,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """
    Run batch inference on test dataset.

    Processes test dataset examples through the model, generating predictions
    and tracking both successful parses and raw outputs.

    Args:
        model: Loaded model (in eval mode)
        tokenizer: Tokenizer for encoding/decoding
        test_dataset: HuggingFace Dataset with test examples
        max_new_tokens: Maximum tokens per generation (default: 512)
        temperature: Sampling temperature (default: 0.1)
        batch_size: Batch size for inference (default: 1 for memory efficiency)
        show_progress: Whether to show progress bar (default: True)

    Returns:
        Tuple of (predictions, ground_truth, raw_outputs):
        - predictions: List of parsed prediction dictionaries
        - ground_truth: List of ground truth intent dictionaries
        - raw_outputs: List of raw generated text strings

    Notes:
        - Predictions list may contain None for failed parses
        - Ground truth extracted from "output" field in dataset
        - Progress bar shows processing speed and ETA
    """
    predictions = []
    ground_truth = []
    raw_outputs = []

    # Create progress bar
    iterator = tqdm(test_dataset, desc="Running inference", disable=not show_progress)  # type: ignore

    for example in iterator:
        # Format prompt (same as training)
        instruction = example.get("instruction", "")
        input_data = example.get("input", "")
        output_data = example.get("output", "")

        prompt = f"{instruction}\n\nInput: {input_data}\n\nOutput:"

        # Run inference
        try:
            generated_text = run_inference(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            raw_outputs.append(generated_text)

            # Parse JSON output
            pred_json = parse_json_output(generated_text)
            predictions.append(pred_json if pred_json else {})

            # Parse ground truth
            truth_json = (
                parse_json_output(output_data)
                if isinstance(output_data, str)
                else output_data
            )
            ground_truth.append(truth_json if truth_json else {})

        except Exception as e:
            logger.warning(f"Inference failed for example: {e}")
            predictions.append({})
            ground_truth.append({})
            raw_outputs.append("")

    return predictions, ground_truth, raw_outputs


def evaluate_model(
    model_path: str | Path,
    test_data_path: str | Path,
    output_path: str | Path | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    tolerance: float = 0.01,
    generate_report: bool = True,
) -> dict[str, Any]:
    """
    Complete evaluation pipeline for fine-tuned model.

    Loads model, runs inference on test set, calculates metrics, and
    optionally generates report.

    Args:
        model_path: Path to fine-tuned model directory
        test_data_path: Path to test.jsonl file
        output_path: Path to save results JSON (optional)
        max_new_tokens: Maximum tokens per generation (default: 512)
        temperature: Sampling temperature (default: 0.1)
        tolerance: Amount accuracy tolerance (default: 0.01 = 1%)
        generate_report: Whether to generate markdown report (default: True)

    Returns:
        Dictionary with evaluation results:
        {
            "overall_accuracy": float,
            "amount_accuracy": float,
            "address_accuracy": float,
            "protocol_accuracy": float,
            "flesch_score": float,
            "per_protocol_metrics": dict,
            "confusion_matrix": list,
            "confusion_matrix_labels": list,
            "total_samples": int,
            "failed_parses": int
        }

    Raises:
        FileNotFoundError: If model or test data not found
        RuntimeError: If evaluation fails

    Notes:
        - Saves results to output_path if provided
        - Generates markdown report if generate_report=True
        - Logs progress and warnings throughout evaluation
    """
    logger.info("=" * 80)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 80)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model_for_evaluation(model_path)

    # Load test dataset
    logger.info(f"Loading test data from {test_data_path}")
    test_data_path = Path(test_data_path)

    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    test_dataset = load_dataset("json", data_files=str(test_data_path))["train"]  # type: ignore
    logger.info(f"Loaded {len(test_dataset)} test examples")  # type: ignore

    # Run inference
    logger.info("Running batch inference...")
    predictions, ground_truth, raw_outputs = batch_inference(
        model,
        tokenizer,
        test_dataset,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Count failed parses
    failed_parses = sum(1 for p in predictions if not p or not isinstance(p, dict))
    logger.info(f"Failed to parse {failed_parses}/{len(predictions)} predictions")

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_accuracy_metrics(predictions, ground_truth, tolerance)

    # Calculate per-protocol metrics
    per_protocol = calculate_per_protocol_metrics(predictions, ground_truth, tolerance)
    metrics["per_protocol_metrics"] = per_protocol
    metrics["failed_parses"] = failed_parses

    # Log results
    logger.info("=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
    logger.info(f"Amount Accuracy: {metrics['amount_accuracy']:.2%}")
    logger.info(f"Address Accuracy: {metrics['address_accuracy']:.2%}")
    logger.info(f"Protocol Accuracy: {metrics['protocol_accuracy']:.2%}")
    if metrics.get("flesch_score", 0) > 0:
        logger.info(f"Flesch Reading Ease: {metrics['flesch_score']:.1f}")
    logger.info(f"Total Samples: {metrics['total_samples']}")
    logger.info(f"Failed Parses: {failed_parses}")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save raw predictions for debugging
        predictions_path = output_path.parent / f"{output_path.stem}_predictions.json"
        with open(predictions_path, "w") as f:
            json.dump(
                {
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                    "raw_outputs": raw_outputs,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved predictions to {predictions_path}")

    # Generate report
    if generate_report and output_path:
        from .report import generate_markdown_report

        report_path = output_path.parent / f"{output_path.stem}_report.md"
        logger.info(f"Generating report: {report_path}")

        generate_markdown_report(
            metrics=metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            output_path=report_path,
        )

    logger.info("=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)

    return metrics
