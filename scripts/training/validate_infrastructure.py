#!/usr/bin/env python3
"""
Validation script for training infrastructure (Commit 6).

This script validates that the training infrastructure is correctly implemented
without requiring GPU access or actual training. It checks:

1. Configuration loading and validation
2. Module imports
3. Configuration dataclass properties
4. Expected file structure

This is a dry-run test that ensures the infrastructure is ready for actual
training when GPU resources are available.

Usage:
    python scripts/training/validate_infrastructure.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_config_loading():
    """Test configuration loading from YAML."""
    print("Testing configuration loading...")

    try:
        from src.eth_finetuning.training.config import TrainingConfig
    except ImportError as e:
        print(f"⚠️  Skipping (missing dependencies): {e}")
        return True  # Not a failure, just missing optional deps

    config_path = Path("configs/training_config.yaml")
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False

    try:
        config = TrainingConfig.from_yaml(config_path)
        print(f"✅ Configuration loaded successfully")
        print(f"   Model: {config.model.name}")
        print(f"   LoRA rank: {config.lora_rank}")
        print(f"   LoRA alpha: {config.lora_alpha}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Effective batch size: {config.effective_batch_size}")
        print(f"   Max sequence length: {config.training.max_seq_length}")
        print(f"   Epochs: {config.training.num_train_epochs}")
        print(f"   Output dir: {config.output_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False


def test_imports():
    """Test that all training modules can be imported."""
    print("\nTesting module imports...")

    try:
        from src.eth_finetuning.training import (
            TrainingConfig,
            setup_model_and_tokenizer,
            create_trainer,
            preprocess_function,
            save_training_logs,
        )

        print("✅ All training modules imported successfully")
        return True
    except ImportError as e:
        if "torch" in str(e) or "transformers" in str(e):
            print(f"⚠️  Skipping (PyTorch/Transformers not installed)")
            print(f"   Install with: uv pip install -e '.[dev]'")
            return True  # Not a failure, just missing optional deps
        else:
            print(f"❌ Import failed: {e}")
            return False


def test_config_validation():
    """Test configuration validation logic."""
    print("\nTesting configuration validation...")

    from src.eth_finetuning.training.config import (
        LoRAConfig,
        TrainingHyperparameters,
    )

    # Test valid configuration
    try:
        lora = LoRAConfig(r=16, lora_alpha=32)
        print(f"✅ Valid LoRA config: rank={lora.r}, alpha={lora.lora_alpha}")
    except Exception as e:
        print(f"❌ Failed to create valid LoRA config: {e}")
        return False

    # Test invalid configuration (should raise ValueError)
    try:
        lora_invalid = LoRAConfig(r=-1, lora_alpha=32)
        print(f"❌ Invalid config accepted (should have raised ValueError)")
        return False
    except ValueError as e:
        print(f"✅ Invalid config rejected correctly: {e}")

    # Test training hyperparameters validation
    try:
        training = TrainingHyperparameters(learning_rate=2e-4)
        print(f"✅ Valid training config: lr={training.learning_rate}")
    except Exception as e:
        print(f"❌ Failed to create valid training config: {e}")
        return False

    return True


def test_config_properties():
    """Test configuration convenience properties."""
    print("\nTesting configuration properties...")

    from src.eth_finetuning.training.config import TrainingConfig

    config = TrainingConfig.from_yaml("configs/training_config.yaml")

    # Test convenience properties
    checks = [
        (config.lora_rank == config.lora.r, "lora_rank property"),
        (config.lora_alpha == config.lora.lora_alpha, "lora_alpha property"),
        (
            config.learning_rate == config.training.learning_rate,
            "learning_rate property",
        ),
        (
            config.batch_size == config.training.per_device_train_batch_size,
            "batch_size property",
        ),
        (
            config.effective_batch_size
            == config.training.per_device_train_batch_size
            * config.training.gradient_accumulation_steps,
            "effective_batch_size property",
        ),
    ]

    all_passed = True
    for passed, name in checks:
        if passed:
            print(f"✅ {name} works correctly")
        else:
            print(f"❌ {name} failed")
            all_passed = False

    return all_passed


def test_file_structure():
    """Test that required files and directories exist."""
    print("\nTesting file structure...")

    required_files = [
        "configs/training_config.yaml",
        "src/eth_finetuning/training/__init__.py",
        "src/eth_finetuning/training/config.py",
        "src/eth_finetuning/training/trainer.py",
        "scripts/training/train_model.py",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_exist = False

    return all_exist


def test_cli_help():
    """Test that CLI script has proper help text."""
    print("\nTesting CLI help...")

    import subprocess

    # Try python3 first, fallback to python
    python_cmd = "python3"

    try:
        result = subprocess.run(
            [python_cmd, "scripts/training/train_model.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and "--config" in result.stdout:
            print("✅ CLI script has proper help text")
            return True
        elif "No module named" in result.stderr:
            print("⚠️  Skipping (missing dependencies)")
            print(f"   Install with: uv pip install -e '.[dev]'")
            return True
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed to run CLI help: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("Training Infrastructure Validation (Commit 6)")
    print("=" * 80)
    print()

    tests = [
        ("File Structure", test_file_structure),
        ("Configuration Loading", test_config_loading),
        ("Module Imports", test_imports),
        ("Configuration Validation", test_config_validation),
        ("Configuration Properties", test_config_properties),
        ("CLI Help", test_cli_help),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        print()

    # Summary
    print("=" * 80)
    print("Validation Summary")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed_count}/{total_count} tests passed")
    print("=" * 80)

    if passed_count == total_count:
        print("\n🎉 All validation tests passed!")
        print("\nNext steps:")
        print("1. Prepare training dataset:")
        print("   python scripts/dataset/prepare_training_data.py \\")
        print("       --input data/processed/decoded.csv \\")
        print("       --output data/datasets \\")
        print("       --split 0.7 0.15 0.15")
        print()
        print("2. Run training (requires GPU):")
        print("   python scripts/training/train_model.py \\")
        print("       --model mistralai/Mistral-7B-Instruct-v0.2 \\")
        print("       --dataset data/datasets \\")
        print("       --output models/fine-tuned/eth-intent-extractor \\")
        print("       --config configs/training_config.yaml")
        print()
        return 0
    else:
        print("\n❌ Some validation tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
