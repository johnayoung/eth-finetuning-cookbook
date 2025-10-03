#!/usr/bin/env python3
"""
CLI script for fine-tuning language models on Ethereum transaction data.

This script provides a command-line interface for training models using QLoRA
on transaction intent extraction tasks. It handles dataset loading, model
setup, training execution, and checkpoint management.

Usage:
    # Basic usage with default config
    python scripts/training/train_model.py \\
        --model mistralai/Mistral-7B-Instruct-v0.2 \\
        --dataset data/datasets \\
        --output models/fine-tuned/eth-intent-extractor \\
        --config configs/training_config.yaml

    # Resume from checkpoint
    python scripts/training/train_model.py \\
        --model mistralai/Mistral-7B-Instruct-v0.2 \\
        --dataset data/datasets \\
        --output models/fine-tuned/eth-intent-extractor \\
        --config configs/training_config.yaml \\
        --resume-from-checkpoint models/fine-tuned/eth-intent-extractor/checkpoint-500

    # Dry run with limited samples
    python scripts/training/train_model.py \\
        --model mistralai/Mistral-7B-Instruct-v0.2 \\
        --dataset data/datasets \\
        --output models/fine-tuned/test-run \\
        --config configs/training_config.yaml \\
        --max-train-samples 10 \\
        --max-eval-samples 5

Requirements:
    - CUDA-capable GPU with 12GB+ VRAM
    - Training dataset in JSONL format (instruction/input/output)
    - Configuration file with QLoRA settings

Hardware Requirements:
    - GPU: RTX 3060 (12GB VRAM) or better
    - RAM: 16GB+ recommended
    - Storage: ~10GB for model cache + checkpoints

Expected Training Time:
    - 1000 samples, 3 epochs: ~2-3 hours on RTX 3060
    - 5000 samples, 3 epochs: ~8-10 hours on RTX 3060
"""

import logging
import sys
from pathlib import Path

import click
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eth_finetuning.training.config import TrainingConfig
from src.eth_finetuning.training.trainer import (
    create_trainer,
    preprocess_function,
    save_training_logs,
    setup_model_and_tokenizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    type=str,
    default=None,
    help="Base model name or path (overrides config file). "
    "Example: mistralai/Mistral-7B-Instruct-v0.2",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True),
    required=True,
    help="Path to dataset directory containing train.jsonl and validation.jsonl",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output directory for fine-tuned model (overrides config file)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to training configuration YAML file",
)
@click.option(
    "--resume-from-checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="Path to checkpoint to resume training from",
)
@click.option(
    "--max-train-samples",
    type=int,
    default=None,
    help="Limit number of training samples (useful for testing)",
)
@click.option(
    "--max-eval-samples",
    type=int,
    default=None,
    help="Limit number of evaluation samples (useful for testing)",
)
@click.option(
    "--no-eval",
    is_flag=True,
    default=False,
    help="Skip evaluation during training (saves time)",
)
def main(
    model: str | None,
    dataset: str,
    output: str | None,
    config: str,
    resume_from_checkpoint: str | None,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    no_eval: bool,
):
    """
    Fine-tune language model on Ethereum transaction data using QLoRA.

    This script loads a pre-trained language model, configures it with
    QLoRA (4-bit quantization + Low-Rank Adaptation), and fine-tunes it
    on transaction intent extraction tasks.

    The training process includes:
    - Loading base model with 4-bit quantization (~7GB VRAM)
    - Applying LoRA adapters to target attention modules
    - Training with gradient accumulation and checkpointing
    - Periodic evaluation and checkpoint saving
    - Final adapter export for inference

    Example:
        python scripts/training/train_model.py \\
            --model mistralai/Mistral-7B-Instruct-v0.2 \\
            --dataset data/datasets \\
            --output models/fine-tuned/eth-intent-extractor \\
            --config configs/training_config.yaml
    """
    logger.info("=" * 80)
    logger.info("Ethereum Intent Extraction - Fine-Tuning Script")
    logger.info("=" * 80)

    # Load configuration
    try:
        train_config = TrainingConfig.from_yaml(config)
        logger.info(f"Configuration loaded from {config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Override config with CLI arguments
    if model:
        train_config.model.name = model
        logger.info(f"Using model from CLI: {model}")

    if output:
        train_config.training.output_dir = output
        logger.info(f"Using output directory from CLI: {output}")

    if resume_from_checkpoint:
        train_config.training.resume_from_checkpoint = resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

    if max_train_samples:
        train_config.dataset.max_train_samples = max_train_samples
        logger.info(f"Limiting training samples to {max_train_samples}")

    if max_eval_samples:
        train_config.dataset.max_eval_samples = max_eval_samples
        logger.info(f"Limiting evaluation samples to {max_eval_samples}")

    if no_eval:
        train_config.training.evaluation_strategy = "no"
        logger.info("Evaluation disabled")

    # Validate dataset paths
    dataset_path = Path(dataset)
    train_file = dataset_path / "train.jsonl"
    val_file = dataset_path / "validation.jsonl"

    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.error("Please run dataset preparation first")
        sys.exit(1)

    logger.info(f"Training file: {train_file}")

    if not no_eval and not val_file.exists():
        logger.warning(f"Validation file not found: {val_file}")
        logger.warning("Continuing without evaluation")
        train_config.training.evaluation_strategy = "no"

    # Load datasets
    logger.info("Loading datasets...")
    try:
        data_files = {"train": str(train_file)}
        if not no_eval and val_file.exists():
            data_files["validation"] = str(val_file)

        raw_datasets = load_dataset("json", data_files=data_files)
        logger.info(f"Loaded {len(raw_datasets['train'])} training examples")  # type: ignore

        if "validation" in raw_datasets:
            logger.info(f"Loaded {len(raw_datasets['validation'])} validation examples")  # type: ignore

    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)

    # Limit samples if specified
    if train_config.dataset.max_train_samples:
        n_samples = min(
            train_config.dataset.max_train_samples,
            len(raw_datasets["train"]),  # type: ignore
        )
        raw_datasets["train"] = raw_datasets["train"].select(range(n_samples))  # type: ignore
        logger.info(f"Limited training samples to {n_samples}")

    if "validation" in raw_datasets and train_config.dataset.max_eval_samples:
        n_samples = min(
            train_config.dataset.max_eval_samples,
            len(raw_datasets["validation"]),  # type: ignore
        )
        raw_datasets["validation"] = raw_datasets["validation"].select(range(n_samples))  # type: ignore
        logger.info(f"Limited validation samples to {n_samples}")

    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    try:
        model, tokenizer = setup_model_and_tokenizer(train_config)  # type: ignore
    except Exception as e:
        logger.error(f"Failed to setup model: {e}")
        logger.error("Ensure you have sufficient VRAM and the model name is correct")
        sys.exit(1)

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    try:
        tokenized_train = raw_datasets["train"].map(  # type: ignore
            lambda examples: preprocess_function(
                examples,
                tokenizer,
                train_config.training.max_seq_length,
                train_config.dataset.padding,
                train_config.dataset.truncation,
            ),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,  # type: ignore
            desc="Tokenizing training data",  # type: ignore
        )

        tokenized_eval = None
        if "validation" in raw_datasets:
            tokenized_eval = raw_datasets["validation"].map(  # type: ignore
                lambda examples: preprocess_function(
                    examples,
                    tokenizer,
                    train_config.training.max_seq_length,
                    train_config.dataset.padding,
                    train_config.dataset.truncation,
                ),
                batched=True,
                remove_columns=raw_datasets["validation"].column_names,  # type: ignore
                desc="Tokenizing validation data",  # type: ignore
            )

        logger.info("Preprocessing completed")

    except Exception as e:
        logger.error(f"Failed to preprocess datasets: {e}")
        sys.exit(1)

    # Create trainer
    logger.info("Creating trainer...")
    try:
        trainer = create_trainer(
            model=model,  # type: ignore
            tokenizer=tokenizer,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            config=train_config,
        )
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        sys.exit(1)

    # Start training
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    try:
        # Check if resuming from checkpoint
        resume_path = train_config.training.resume_from_checkpoint
        if resume_path:
            logger.info(f"Resuming from checkpoint: {resume_path}")

        # Train
        train_result = trainer.train(resume_from_checkpoint=resume_path)

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

        # Log training metrics
        metrics = train_result.metrics
        logger.info("Training Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_model()
        sys.exit(130)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Attempting to save checkpoint...")
        try:
            trainer.save_model()
            logger.info("Checkpoint saved successfully")
        except:
            logger.error("Failed to save checkpoint")
        sys.exit(1)

    # Save final model
    logger.info("Saving final model and adapters...")
    try:
        output_dir = Path(train_config.training.output_dir)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))  # type: ignore

        logger.info(f"Model saved to {output_dir}")
        logger.info("Files saved:")
        logger.info("  - adapter_model.bin: LoRA adapter weights")
        logger.info("  - adapter_config.json: LoRA configuration")
        logger.info("  - tokenizer_config.json: Tokenizer configuration")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)

    # Save training logs
    logger.info("Saving training logs...")
    try:
        log_path = output_dir / "training_logs.txt"
        save_training_logs(trainer, log_path)
        logger.info(f"Training logs saved to {log_path}")
    except Exception as e:
        logger.warning(f"Failed to save training logs: {e}")

    # Final summary
    logger.info("=" * 80)
    logger.info("Fine-Tuning Complete!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training samples: {len(tokenized_train)}")
    if tokenized_eval:
        logger.info(f"Validation samples: {len(tokenized_eval)}")
    logger.info(f"Epochs: {train_config.training.num_train_epochs}")
    logger.info(f"Effective batch size: {train_config.effective_batch_size}")
    logger.info("=" * 80)

    logger.info("\nNext steps:")
    logger.info("1. Run evaluation:")
    logger.info(
        f"   python scripts/evaluation/evaluate_model.py --model {output_dir} "
        f"--test-data {dataset_path / 'test.jsonl'}"
    )
    logger.info("2. Test inference:")
    logger.info(
        f"   python scripts/examples/run_inference.py --model {output_dir} "
        "--tx-hash <transaction_hash>"
    )


if __name__ == "__main__":
    main()
