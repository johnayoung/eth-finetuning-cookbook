"""
Training configuration management for fine-tuning pipeline.

This module provides utilities for loading, validating, and accessing
training configuration from YAML files. It ensures type-safe access to
hyperparameters and provides sensible defaults.

Usage:
    from eth_finetuning.training.config import TrainingConfig

    config = TrainingConfig.from_yaml("configs/training_config.yaml")
    print(config.lora_rank)  # 16
    print(config.learning_rate)  # 0.0002
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Base model configuration."""

    name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    cache_dir: str = "models/base"
    trust_remote_code: bool = False


@dataclass
class QuantizationConfig:
    """4-bit quantization configuration for QLoRA."""

    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""

    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha scaling factor
    lora_dropout: float = 0.05  # Dropout for LoRA layers
    bias: str = "none"  # Bias parameter handling
    task_type: str = "CAUSAL_LM"  # Task type for PEFT
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def __post_init__(self):
        """Validate configuration values."""
        if self.r <= 0:
            raise ValueError(f"LoRA rank (r) must be positive, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.lora_alpha}")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"LoRA dropout must be in [0, 1], got {self.lora_dropout}")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(
                f"bias must be 'none', 'all', or 'lora_only', got {self.bias}"
            )
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")


@dataclass
class TrainingHyperparameters:
    """Training hyperparameters."""

    # Learning rate
    learning_rate: float = 2e-4

    # Batch configuration
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1

    # Sequence length
    max_seq_length: int = 2048

    # Optimization
    optim: str = "paged_adamw_32bit"
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Gradient optimization
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True

    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3

    # Output
    output_dir: str = "models/fine-tuned/eth-intent-extractor"
    overwrite_output_dir: bool = False

    # Reproducibility
    seed: int = 42

    # Performance
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Reporting
    report_to: list[str] = field(default_factory=list)

    # Checkpoint recovery
    resume_from_checkpoint: str | None = None

    def __post_init__(self):
        """Validate hyperparameter values."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.per_device_train_batch_size <= 0:
            raise ValueError(
                f"per_device_train_batch_size must be positive, "
                f"got {self.per_device_train_batch_size}"
            )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.num_train_epochs <= 0 and self.max_steps <= 0:
            raise ValueError("Either num_train_epochs or max_steps must be positive")
        if self.max_seq_length <= 0:
            raise ValueError(
                f"max_seq_length must be positive, got {self.max_seq_length}"
            )
        if self.evaluation_strategy not in ["no", "steps", "epoch"]:
            raise ValueError(
                f"evaluation_strategy must be 'no', 'steps', or 'epoch', "
                f"got {self.evaluation_strategy}"
            )
        if self.save_strategy not in ["no", "steps", "epoch"]:
            raise ValueError(
                f"save_strategy must be 'no', 'steps', or 'epoch', "
                f"got {self.save_strategy}"
            )


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    train_file: str = "data/datasets/train.jsonl"
    validation_file: str = "data/datasets/validation.jsonl"
    test_file: str = "data/datasets/test.jsonl"

    # Data preprocessing
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Tokenization
    padding: str = "max_length"
    truncation: bool = True
    add_special_tokens: bool = True

    def __post_init__(self):
        """Validate dataset configuration."""
        if self.padding not in ["max_length", "longest", "do_not_pad"]:
            raise ValueError(
                f"padding must be 'max_length', 'longest', or 'do_not_pad', "
                f"got {self.padding}"
            )


@dataclass
class HardwareConfig:
    """Hardware configuration."""

    use_cuda: bool = True
    device_map: str = "auto"
    low_cpu_mem_usage: bool = True


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # Convenience properties for common access patterns
    @property
    def lora_rank(self) -> int:
        """Get LoRA rank."""
        return self.lora.r

    @property
    def lora_alpha(self) -> int:
        """Get LoRA alpha."""
        return self.lora.lora_alpha

    @property
    def learning_rate(self) -> float:
        """Get learning rate."""
        return self.training.learning_rate

    @property
    def batch_size(self) -> int:
        """Get per-device training batch size."""
        return self.training.per_device_train_batch_size

    @property
    def effective_batch_size(self) -> int:
        """Get effective batch size (batch_size * gradient_accumulation_steps)."""
        return (
            self.training.per_device_train_batch_size
            * self.training.gradient_accumulation_steps
        )

    @property
    def output_dir(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.training.output_dir)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainingConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            TrainingConfig instance with loaded values

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML structure is invalid
            yaml.YAMLError: If YAML parsing fails
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        logger.info(f"Loading training configuration from {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if not config_dict:
            raise ValueError(f"Empty or invalid YAML file: {yaml_path}")

        # Parse each section
        try:
            model_config = ModelConfig(**config_dict.get("model", {}))
            quant_config = QuantizationConfig(**config_dict.get("quantization", {}))
            lora_config = LoRAConfig(**config_dict.get("lora", {}))
            training_config = TrainingHyperparameters(**config_dict.get("training", {}))
            dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
            hardware_config = HardwareConfig(**config_dict.get("hardware", {}))

            config = cls(
                model=model_config,
                quantization=quant_config,
                lora=lora_config,
                training=training_config,
                dataset=dataset_config,
                hardware=hardware_config,
            )

            logger.info("Training configuration loaded successfully")
            logger.debug(f"Model: {config.model.name}")
            logger.debug(f"LoRA rank: {config.lora_rank}")
            logger.debug(f"Learning rate: {config.learning_rate}")
            logger.debug(f"Effective batch size: {config.effective_batch_size}")

            return config

        except TypeError as e:
            raise ValueError(
                f"Invalid configuration structure in {yaml_path}: {e}"
            ) from e

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            TrainingConfig instance
        """
        model_config = ModelConfig(**config_dict.get("model", {}))
        quant_config = QuantizationConfig(**config_dict.get("quantization", {}))
        lora_config = LoRAConfig(**config_dict.get("lora", {}))
        training_config = TrainingHyperparameters(**config_dict.get("training", {}))
        dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
        hardware_config = HardwareConfig(**config_dict.get("hardware", {}))

        return cls(
            model=model_config,
            quantization=quant_config,
            lora=lora_config,
            training=training_config,
            dataset=dataset_config,
            hardware=hardware_config,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "model": {
                "name": self.model.name,
                "cache_dir": self.model.cache_dir,
                "trust_remote_code": self.model.trust_remote_code,
            },
            "quantization": {
                "load_in_4bit": self.quantization.load_in_4bit,
                "bnb_4bit_compute_dtype": self.quantization.bnb_4bit_compute_dtype,
                "bnb_4bit_quant_type": self.quantization.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": self.quantization.bnb_4bit_use_double_quant,
            },
            "lora": {
                "r": self.lora.r,
                "lora_alpha": self.lora.lora_alpha,
                "lora_dropout": self.lora.lora_dropout,
                "bias": self.lora.bias,
                "task_type": self.lora.task_type,
                "target_modules": self.lora.target_modules,
            },
            "training": {
                "learning_rate": self.training.learning_rate,
                "per_device_train_batch_size": self.training.per_device_train_batch_size,
                "per_device_eval_batch_size": self.training.per_device_eval_batch_size,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "num_train_epochs": self.training.num_train_epochs,
                "max_steps": self.training.max_steps,
                "max_seq_length": self.training.max_seq_length,
                "optim": self.training.optim,
                "weight_decay": self.training.weight_decay,
                "warmup_steps": self.training.warmup_steps,
                "lr_scheduler_type": self.training.lr_scheduler_type,
                "fp16": self.training.fp16,
                "bf16": self.training.bf16,
                "gradient_checkpointing": self.training.gradient_checkpointing,
                "max_grad_norm": self.training.max_grad_norm,
                "logging_steps": self.training.logging_steps,
                "logging_first_step": self.training.logging_first_step,
                "evaluation_strategy": self.training.evaluation_strategy,
                "eval_steps": self.training.eval_steps,
                "save_strategy": self.training.save_strategy,
                "save_steps": self.training.save_steps,
                "save_total_limit": self.training.save_total_limit,
                "output_dir": self.training.output_dir,
                "overwrite_output_dir": self.training.overwrite_output_dir,
                "seed": self.training.seed,
                "dataloader_num_workers": self.training.dataloader_num_workers,
                "dataloader_pin_memory": self.training.dataloader_pin_memory,
                "report_to": self.training.report_to,
                "resume_from_checkpoint": self.training.resume_from_checkpoint,
            },
            "dataset": {
                "train_file": self.dataset.train_file,
                "validation_file": self.dataset.validation_file,
                "test_file": self.dataset.test_file,
                "max_train_samples": self.dataset.max_train_samples,
                "max_eval_samples": self.dataset.max_eval_samples,
                "padding": self.dataset.padding,
                "truncation": self.dataset.truncation,
                "add_special_tokens": self.dataset.add_special_tokens,
            },
            "hardware": {
                "use_cuda": self.hardware.use_cuda,
                "device_map": self.hardware.device_map,
                "low_cpu_mem_usage": self.hardware.low_cpu_mem_usage,
            },
        }

    def save_yaml(self, yaml_path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {yaml_path}")
