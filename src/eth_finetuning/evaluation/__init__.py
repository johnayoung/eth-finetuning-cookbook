"""
Evaluation module for fine-tuned Ethereum intent extraction models.

This module provides utilities for quantifying model performance against
accuracy and readability targets. It includes metrics calculation,
confusion matrix generation, and report creation.

Main Components:
    - evaluator: Model loading and batch inference logic
    - metrics: Accuracy calculations and readability scoring
    - report: Human-readable report generation

Usage:
    from eth_finetuning.evaluation import evaluate_model, calculate_accuracy_metrics

    results = evaluate_model(model_path, test_data_path)
    metrics = calculate_accuracy_metrics(predictions, ground_truth)
"""

from .evaluator import evaluate_model, load_model_for_evaluation
from .metrics import (
    calculate_accuracy_metrics,
    calculate_address_accuracy,
    calculate_amount_accuracy,
    calculate_flesch_score,
    calculate_per_protocol_metrics,
    calculate_protocol_accuracy,
    create_confusion_matrix,
)
from .report import generate_markdown_report

__all__ = [
    "evaluate_model",
    "load_model_for_evaluation",
    "calculate_accuracy_metrics",
    "calculate_amount_accuracy",
    "calculate_address_accuracy",
    "calculate_protocol_accuracy",
    "calculate_flesch_score",
    "calculate_per_protocol_metrics",
    "create_confusion_matrix",
    "generate_markdown_report",
]
