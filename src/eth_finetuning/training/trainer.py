"""
Fine-tuning pipeline with QLoRA for Ethereum intent extraction.

This module provides the core training logic for fine-tuning language models
on transaction data using QLoRA (Quantized Low-Rank Adaptation). It handles
model loading with 4-bit quantization, LoRA adapter configuration, and
HuggingFace Trainer setup optimized for 12GB VRAM GPUs.

Usage:
    from eth_finetuning.training.trainer import setup_model_and_tokenizer, create_trainer
    from eth_finetuning.training.config import TrainingConfig

    config = TrainingConfig.from_yaml("configs/training_config.yaml")
    model, tokenizer = setup_model_and_tokenizer(config)
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)
    trainer.train()
"""

import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from .config import TrainingConfig

logger = logging.getLogger(__name__)


def get_compute_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert dtype string to torch dtype.

    Args:
        dtype_str: String representation of dtype ("float16", "bfloat16", "float32")

    Returns:
        Corresponding torch dtype

    Raises:
        ValueError: If dtype_str is not recognized
    """
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    dtype = dtype_map.get(dtype_str.lower())
    if dtype is None:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. " f"Supported: {list(dtype_map.keys())}"
        )

    return dtype


def setup_model_and_tokenizer(
    config: TrainingConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and configure base model with 4-bit quantization and LoRA adapters.

    This function performs the following steps:
    1. Configure 4-bit quantization (BitsAndBytes)
    2. Load base model with quantization
    3. Load tokenizer and configure padding
    4. Prepare model for k-bit training
    5. Apply LoRA adapters to target modules
    6. Enable gradient checkpointing if configured

    Args:
        config: Training configuration with model, quantization, and LoRA settings

    Returns:
        Tuple of (model, tokenizer) ready for training

    Raises:
        RuntimeError: If model loading or PEFT configuration fails
        ValueError: If configuration values are invalid

    Notes:
        - Model is loaded in 4-bit to reduce VRAM from ~28GB to ~7GB
        - LoRA adapters add ~0.5GB VRAM
        - Gradient checkpointing trades compute for memory (~30% VRAM savings)
        - Total VRAM usage: ~11-12GB (safe for 12GB GPUs)
    """
    logger.info("Setting up model and tokenizer...")
    logger.info(f"Base model: {config.model.name}")
    logger.info(f"LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
    logger.info(
        f"Effective batch size: {config.effective_batch_size} "
        f"({config.batch_size} * {config.training.gradient_accumulation_steps})"
    )

    # Step 1: Configure 4-bit quantization
    compute_dtype = get_compute_dtype(config.quantization.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )

    logger.info("4-bit quantization configured:")
    logger.info(f"  Quant type: {config.quantization.bnb_4bit_quant_type}")
    logger.info(f"  Compute dtype: {config.quantization.bnb_4bit_compute_dtype}")
    logger.info(f"  Double quant: {config.quantization.bnb_4bit_use_double_quant}")

    # Step 2: Load base model with quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            quantization_config=bnb_config,
            device_map=config.hardware.device_map,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir,
            low_cpu_mem_usage=config.hardware.low_cpu_mem_usage,
        )
        logger.info("Base model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model {config.model.name}: {e}")
        raise RuntimeError(
            f"Model loading failed. Ensure model name is correct and "
            f"you have sufficient memory. Error: {e}"
        ) from e

    # Step 3: Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir,
        )

        # Configure padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Set pad_token to eos_token")

        # Set padding side to left for decoder-only models
        tokenizer.padding_side = "left"

        logger.info("Tokenizer loaded successfully")
        logger.info(f"  Vocab size: {len(tokenizer)}")
        logger.info(
            f"  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})"
        )

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise RuntimeError(f"Tokenizer loading failed: {e}") from e

    # Step 4: Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
    )
    logger.info("Model prepared for k-bit training")

    # Step 5: Configure and apply LoRA
    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,  # type: ignore
        task_type=config.lora.task_type,
        target_modules=config.lora.target_modules,
    )

    try:
        model = get_peft_model(model, peft_config)
        logger.info("LoRA adapters applied successfully")
    except Exception as e:
        logger.error(f"Failed to apply LoRA adapters: {e}")
        raise RuntimeError(f"PEFT configuration failed: {e}") from e

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")

    # Step 6: Enable gradient checkpointing if configured
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore
        logger.info("Gradient checkpointing enabled (saves ~30% VRAM)")

    return model, tokenizer  # type: ignore


def preprocess_function(
    examples: dict[str, list[Any]],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    padding: str = "max_length",
    truncation: bool = True,
) -> dict[str, list[Any]]:
    """
    Tokenize and preprocess training examples.

    Converts text inputs to token IDs and creates attention masks.
    Combines instruction, input, and output into a single sequence.

    Args:
        examples: Batch of examples with 'instruction', 'input', 'output' keys
        tokenizer: Tokenizer for encoding text
        max_seq_length: Maximum sequence length
        padding: Padding strategy ("max_length" or "longest")
        truncation: Whether to truncate sequences exceeding max_length

    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'labels'

    Notes:
        - Format: "<instruction>\n\nInput: <input>\n\nOutput: <output>"
        - Labels are set to input_ids (causal LM training)
        - Padding tokens in labels are set to -100 (ignored in loss)
    """
    # Combine instruction, input, and output into prompt format
    prompts = []
    for instruction, inp, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        # Format: Instruction + Input + Output
        prompt = f"{instruction}\n\nInput: {inp}\n\nOutput: {output}"
        prompts.append(prompt)

    # Tokenize
    tokenized = tokenizer(  # type: ignore
        prompts,
        padding=padding,
        truncation=truncation,
        max_length=max_seq_length,
        return_tensors=None,  # Return lists for datasets
    )

    # Set labels for causal LM (same as input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()

    # Set padding tokens in labels to -100 (ignored in loss calculation)
    if tokenizer.pad_token_id is not None:  # type: ignore
        for i, label_seq in enumerate(tokenized["labels"]):
            tokenized["labels"][i] = [
                -100 if token_id == tokenizer.pad_token_id else token_id  # type: ignore
                for token_id in label_seq
            ]

    return tokenized


def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    config: TrainingConfig,
) -> Trainer:
    """
    Create HuggingFace Trainer with optimized settings for 12GB VRAM.

    Configures training arguments including:
    - Batch size and gradient accumulation
    - Mixed precision training (fp16/bf16)
    - Learning rate schedule with warmup
    - Checkpointing strategy
    - Logging and evaluation intervals

    Args:
        model: Model with LoRA adapters (from setup_model_and_tokenizer)
        tokenizer: Tokenizer for preprocessing
        train_dataset: Training dataset (HuggingFace Dataset)
        eval_dataset: Validation dataset (optional)
        config: Training configuration

    Returns:
        Configured Trainer ready for training

    Raises:
        ValueError: If datasets are empty or invalid

    Notes:
        - Checkpoints saved every save_steps (default: 500)
        - Evaluation run every eval_steps (default: 500)
        - Only keeps last save_total_limit checkpoints (default: 3)
    """
    if not train_dataset:
        raise ValueError("train_dataset cannot be empty")

    logger.info("Creating Trainer...")
    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation samples: {len(eval_dataset)}")

    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure training arguments
    # type: ignore - TrainingArguments parameters are dynamic
    training_args = TrainingArguments(  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue,reportArgumentType]
        # Output
        output_dir=str(output_dir),
        overwrite_output_dir=config.training.overwrite_output_dir,
        # Training duration
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        # Batch size
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        # Optimization
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optim=config.training.optim,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        # Mixed precision
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        # Gradient optimization
        gradient_checkpointing=config.training.gradient_checkpointing,
        max_grad_norm=config.training.max_grad_norm,
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=config.training.logging_steps,
        logging_first_step=config.training.logging_first_step,
        # Evaluation
        eval_strategy=config.training.evaluation_strategy,  # Renamed from evaluation_strategy
        eval_steps=config.training.eval_steps if eval_dataset else None,
        # Checkpointing
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        # Performance
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        # Reproducibility
        seed=config.training.seed,
        # Reporting
        report_to=config.training.report_to,
        # Misc
        remove_unused_columns=False,  # Keep all dataset columns
        load_best_model_at_end=False,  # Don't reload best model (saves memory)
    )

    # Create trainer
    # type: ignore - Trainer parameters are dynamic
    trainer = Trainer(  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue,reportArgumentType]
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # type: ignore
    )

    logger.info("Trainer created successfully")
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Effective batch size: {config.effective_batch_size} "
        f"({config.batch_size} * {config.training.gradient_accumulation_steps})"
    )
    logger.info(
        f"Total training steps: ~{len(train_dataset) * config.training.num_train_epochs // config.effective_batch_size}"
    )

    return trainer


def save_training_logs(
    trainer: Trainer,
    output_path: str | Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """
    Save training logs to text file.

    Extracts training history from trainer state and saves to human-readable
    format with loss, learning rate, VRAM usage, and training time.

    Args:
        trainer: Trained Trainer instance
        output_path: Path to save logs (e.g., "training_logs.txt")
        start_time: Training start timestamp (from time.time())
        end_time: Training end timestamp (from time.time())

    Notes:
        - Logs include step number, loss, learning rate
        - Includes peak VRAM usage if CUDA is available
        - Includes total training time if timestamps provided
        - Useful for plotting training curves and performance analysis
        - Saved in addition to HuggingFace's trainer_state.json
    """
    import time

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving training logs to {output_path}")

    with open(output_path, "w") as f:
        f.write("Training Logs\n")
        f.write("=" * 80 + "\n\n")

        # System information
        f.write("System Information:\n")
        f.write("-" * 80 + "\n")

        # CUDA/GPU information
        if torch.cuda.is_available():
            f.write(f"CUDA Available: Yes\n")
            if hasattr(torch.version, "cuda") and torch.version.cuda:  # type: ignore
                f.write(f"CUDA Version: {torch.version.cuda}\n")  # type: ignore
            f.write(f"GPU Device: {torch.cuda.get_device_name(0)}\n")

            # Get peak memory usage
            peak_memory_allocated = torch.cuda.max_memory_allocated(0) / 1024**3  # GB
            peak_memory_reserved = torch.cuda.max_memory_reserved(0) / 1024**3  # GB

            f.write(f"Peak VRAM Allocated: {peak_memory_allocated:.2f} GB\n")
            f.write(f"Peak VRAM Reserved: {peak_memory_reserved:.2f} GB\n")
        else:
            f.write(f"CUDA Available: No (CPU training)\n")

        f.write("\n")

        # Training time
        if start_time and end_time:
            total_time = end_time - start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            f.write(
                f"Total Training Time: {hours}h {minutes}m {seconds}s ({total_time:.2f}s)\n"
            )
            f.write("\n")

        # Training history
        if trainer.state.log_history:
            f.write("Training History:\n")
            f.write("-" * 80 + "\n")
            f.write("Step\tLoss\tLearning Rate\n")
            f.write("-" * 80 + "\n")

            for log_entry in trainer.state.log_history:
                step = log_entry.get("step", "N/A")
                loss = log_entry.get("loss", log_entry.get("eval_loss", "N/A"))
                lr = log_entry.get("learning_rate", "N/A")

                f.write(f"{step}\t{loss}\t{lr}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Training Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total FLOPs: {trainer.state.total_flos}\n")
        f.write(f"Best checkpoint: {trainer.state.best_model_checkpoint}\n")

        # Final epoch/step info
        if trainer.state.log_history:
            final_entry = trainer.state.log_history[-1]
            if "epoch" in final_entry:
                f.write(f"Final epoch: {final_entry['epoch']}\n")
            if "step" in final_entry:
                f.write(f"Final step: {final_entry['step']}\n")

    logger.info("Training logs saved successfully")


def load_model_for_inference(
    model_path: str | Path,
    adapter_path: str | Path,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load fine-tuned model with adapter for inference.

    Args:
        model_path: Path to base model
        adapter_path: Path to fine-tuned LoRA adapter
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer) ready for inference

    Notes:
        - Model is loaded in 4-bit by default for memory efficiency
        - Use model.eval() and torch.no_grad() for inference
    """
    from peft import PeftModel

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Loading adapter from {adapter_path}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=False,
    )

    # Load adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore

    logger.info("Model and adapter loaded successfully")

    return model, tokenizer  # type: ignore
