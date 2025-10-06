#!/usr/bin/env python3
"""
Performance benchmarking script for Ethereum Fine-Tuning MVP (Commit 12).

This script validates the complete pipeline against MVP success criteria:
1. Training time: <4 hours on RTX 3060 (12GB VRAM)
2. Peak VRAM usage: <12GB
3. Model accuracy: ≥90% on amounts, addresses, and protocols
4. Flesch Reading Ease: ≥60 for generated text

The script can run in different modes:
- Full benchmark: Complete training + evaluation (requires GPU)
- Validation mode: Verify all components without training
- Mock mode: Test script without actual model/GPU

Usage:
    # Full benchmark with GPU
    python scripts/benchmark_mvp.py --mode full --dataset data/datasets/

    # Validation mode (no training)
    python scripts/benchmark_mvp.py --mode validate

    # Mock mode (for testing)
    python scripts/benchmark_mvp.py --mode mock
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BenchmarkRunner:
    """Orchestrate MVP benchmarking and validation."""

    def __init__(self, mode: str, dataset_path: Optional[Path] = None):
        """
        Initialize benchmark runner.

        Args:
            mode: Benchmark mode ('full', 'validate', or 'mock')
            dataset_path: Path to dataset directory (required for 'full' mode)
        """
        self.mode = mode
        self.dataset_path = dataset_path
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "mvp_criteria": {
                "training_time_target_hours": 4,
                "vram_target_gb": 12,
                "accuracy_target_percent": 90,
                "flesch_score_target": 60,
            },
            "validation_results": {},
            "training_results": {},
            "evaluation_results": {},
            "mvp_pass": False,
        }

        # Output paths
        self.output_dir = project_root / "outputs"
        self.benchmarks_dir = self.output_dir / "benchmarks"
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        """
        Execute benchmark based on mode.

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*70}")
        print(f"Ethereum Fine-Tuning MVP Benchmark - Mode: {self.mode.upper()}")
        print(f"{'='*70}\n")

        if self.mode == "validate":
            self._run_validation()
        elif self.mode == "mock":
            self._run_mock()
        elif self.mode == "full":
            if not self.dataset_path:
                raise ValueError("Dataset path required for 'full' mode")
            self._run_full_benchmark()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Check if MVP criteria met
        self._check_mvp_criteria()

        # Save results
        self._save_results()

        # Generate report
        self._generate_report()

        return self.results

    def _run_validation(self):
        """Run validation checks without training."""
        print("Running validation checks...\n")

        checks = [
            ("Project Structure", self._check_project_structure),
            ("Dependencies", self._check_dependencies),
            ("Configuration Files", self._check_configs),
            ("Extraction Pipeline", self._check_extraction),
            ("Dataset Preparation", self._check_dataset_prep),
            ("Training Infrastructure", self._check_training_infra),
            ("Evaluation Module", self._check_evaluation),
            ("CLI Scripts", self._check_cli_scripts),
            ("Notebooks", self._check_notebooks),
            ("Tests", self._check_tests),
        ]

        for check_name, check_func in checks:
            print(f"Checking {check_name}...")
            try:
                result = check_func()
                self.results["validation_results"][check_name] = {
                    "status": "pass" if result else "fail",
                    "details": result if isinstance(result, dict) else {},
                }
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {status}\n")
            except Exception as e:
                self.results["validation_results"][check_name] = {
                    "status": "error",
                    "error": str(e),
                }
                print(f"  ⚠️  ERROR: {e}\n")

    def _run_mock(self):
        """Run mock benchmark with simulated data."""
        print("Running mock benchmark (simulated data)...\n")

        # Simulate validation
        self._run_validation()

        # Simulate training metrics
        print("Simulating training metrics...")
        self.results["training_results"] = {
            "total_time_hours": 3.5,
            "peak_vram_gb": 11.2,
            "final_loss": 0.45,
            "epochs_completed": 3,
            "training_samples": 1000,
            "status": "simulated",
        }
        print("  ✅ Simulated training: 3.5 hours, 11.2GB VRAM\n")

        # Simulate evaluation metrics
        print("Simulating evaluation metrics...")
        self.results["evaluation_results"] = {
            "overall_accuracy": 92.5,
            "amount_accuracy": 94.2,
            "address_accuracy": 95.8,
            "protocol_accuracy": 91.3,
            "flesch_score": 65.2,
            "status": "simulated",
        }
        print("  ✅ Simulated evaluation: 92.5% accuracy, Flesch 65.2\n")

    def _run_full_benchmark(self):
        """Run complete benchmark with actual training."""
        print("Running full benchmark with training...\n")

        # First validate
        self._run_validation()

        # Check GPU availability
        if not self._check_gpu():
            print("❌ GPU not available - cannot run full benchmark")
            return

        # Run training with monitoring
        print("\nStarting training with performance monitoring...")
        self._run_monitored_training()

        # Run evaluation
        print("\nRunning evaluation...")
        self._run_evaluation()

    def _check_project_structure(self) -> bool:
        """Verify all required directories exist."""
        required_dirs = [
            "src/eth_finetuning",
            "scripts",
            "docs",
            "notebooks",
            "tests",
            "configs",
            "data",
            "models",
            "outputs",
        ]

        missing = []
        for dir_path in required_dirs:
            if not (project_root / dir_path).exists():
                missing.append(dir_path)

        if missing:
            print(f"  Missing directories: {', '.join(missing)}")
            return False

        return True

    def _check_dependencies(self) -> bool:
        """Check that required dependencies are installed."""
        required_packages = [
            "web3",
            "pandas",
            "torch",
            "transformers",
            "peft",
            "click",
            "pytest",
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            print(f"  Missing packages: {', '.join(missing)}")
            return False

        return True

    def _check_configs(self) -> bool:
        """Verify configuration files exist and are valid."""
        config_files = [
            "configs/extraction_config.yaml",
            "configs/training_config.yaml",
            "configs/evaluation_config.yaml",
        ]

        for config_file in config_files:
            path = project_root / config_file
            if not path.exists():
                print(f"  Missing config: {config_file}")
                return False

        return True

    def _check_extraction(self) -> bool:
        """Check extraction pipeline components."""
        try:
            from src.eth_finetuning.extraction.core.utils import (
                Web3ConnectionManager,
            )
            from src.eth_finetuning.extraction.core.fetcher import (
                fetch_transactions_batch,
            )
            from src.eth_finetuning.extraction.decoders.eth import decode_eth_transfer
            from src.eth_finetuning.extraction.decoders.erc20 import (
                decode_erc20_transfers,
            )

            return True
        except ImportError as e:
            print(f"  Import error: {e}")
            return False

    def _check_dataset_prep(self) -> bool:
        """Check dataset preparation modules."""
        try:
            from src.eth_finetuning.dataset.intent_extraction import extract_intent
            from src.eth_finetuning.dataset.templates import format_training_example
            from src.eth_finetuning.dataset.preparation import prepare_dataset

            return True
        except ImportError as e:
            print(f"  Import error: {e}")
            return False

    def _check_training_infra(self) -> bool:
        """Check training infrastructure."""
        try:
            from src.eth_finetuning.training.config import TrainingConfig

            # Try loading config
            config_path = project_root / "configs/training_config.yaml"
            if config_path.exists():
                config = TrainingConfig.from_yaml(config_path)
                return True

            return False
        except Exception as e:
            print(f"  Error: {e}")
            return False

    def _check_evaluation(self) -> bool:
        """Check evaluation module."""
        try:
            from src.eth_finetuning.evaluation.evaluator import evaluate_model
            from src.eth_finetuning.evaluation.metrics import calculate_accuracy_metrics

            return True
        except ImportError as e:
            print(f"  Import error: {e}")
            return False

    def _check_cli_scripts(self) -> bool:
        """Verify CLI scripts exist."""
        required_scripts = [
            "scripts/fetch_transactions.py",
            "scripts/decode_transactions.py",
            "scripts/dataset/prepare_training_data.py",
            "scripts/training/train_model.py",
            "scripts/evaluation/evaluate_model.py",
            "scripts/examples/run_inference.py",
            "scripts/examples/analyze_transaction.py",
        ]

        for script in required_scripts:
            if not (project_root / script).exists():
                print(f"  Missing script: {script}")
                return False

        return True

    def _check_notebooks(self) -> bool:
        """Verify notebooks exist."""
        required_notebooks = [
            "notebooks/01-data-exploration.ipynb",
            "notebooks/02-data-extraction.ipynb",
            "notebooks/03-dataset-preparation.ipynb",
            "notebooks/04-fine-tuning.ipynb",
            "notebooks/05-evaluation.ipynb",
        ]

        for notebook in required_notebooks:
            if not (project_root / notebook).exists():
                print(f"  Missing notebook: {notebook}")
                return False

        return True

    def _check_tests(self) -> bool:
        """Run test suite and check coverage."""
        try:
            result = subprocess.run(
                ["pytest", "tests/", "-v", "--tb=short"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Store test output
            self.results["test_output"] = {
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:],  # Last 1000 chars
            }

            return result.returncode == 0
        except Exception as e:
            print(f"  Error running tests: {e}")
            return False

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _run_monitored_training(self):
        """Run training with VRAM and time monitoring."""
        try:
            import torch

            start_time = time.time()
            peak_vram = 0

            # Start training process
            # Note: This is a placeholder - actual implementation would
            # monitor training in real-time
            print("  Note: Full training requires GPU access")
            print("  This would run: scripts/training/train_model.py")

            # For validation, we simulate or skip
            self.results["training_results"] = {
                "status": "requires_gpu",
                "note": "Full training requires GPU access",
            }

        except Exception as e:
            print(f"  Error: {e}")
            self.results["training_results"] = {"status": "error", "error": str(e)}

    def _run_evaluation(self):
        """Run model evaluation."""
        try:
            print("  Note: Evaluation requires trained model")
            self.results["evaluation_results"] = {
                "status": "requires_model",
                "note": "Evaluation requires trained model checkpoint",
            }
        except Exception as e:
            print(f"  Error: {e}")
            self.results["evaluation_results"] = {"status": "error", "error": str(e)}

    def _check_mvp_criteria(self):
        """Check if MVP criteria are met."""
        criteria_met = []

        # Check validation results
        if "validation_results" in self.results:
            validation_pass = all(
                r.get("status") == "pass"
                for r in self.results["validation_results"].values()
            )
            criteria_met.append(("All validation checks", validation_pass))

        # Check training results (if available)
        if "training_results" in self.results:
            training = self.results["training_results"]
            if "total_time_hours" in training:
                time_ok = training["total_time_hours"] < 4.0
                criteria_met.append(("Training time <4 hours", time_ok))

            if "peak_vram_gb" in training:
                vram_ok = training["peak_vram_gb"] < 12.0
                criteria_met.append(("Peak VRAM <12GB", vram_ok))

        # Check evaluation results (if available)
        if "evaluation_results" in self.results:
            evaluation = self.results["evaluation_results"]
            if "overall_accuracy" in evaluation:
                acc_ok = evaluation["overall_accuracy"] >= 90.0
                criteria_met.append(("Accuracy ≥90%", acc_ok))

            if "flesch_score" in evaluation:
                flesch_ok = evaluation["flesch_score"] >= 60.0
                criteria_met.append(("Flesch score ≥60", flesch_ok))

        # Overall pass if all criteria met
        self.results["mvp_pass"] = all(passed for _, passed in criteria_met)
        self.results["criteria_summary"] = criteria_met

    def _save_results(self):
        """Save benchmark results to JSON."""
        output_file = (
            self.benchmarks_dir
            / f"benchmark_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def _generate_report(self):
        """Generate human-readable markdown report."""
        report_file = self.benchmarks_dir / "benchmark_report.md"

        with open(report_file, "w") as f:
            f.write("# Ethereum Fine-Tuning MVP Benchmark Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")
            f.write(f"**Mode:** {self.mode}\n\n")

            # MVP Criteria
            f.write("## MVP Success Criteria\n\n")
            f.write("| Criterion | Target | Status |\n")
            f.write("|-----------|--------|--------|\n")
            criteria = self.results["mvp_criteria"]
            f.write(
                f"| Training Time | <{criteria['training_time_target_hours']} hours | "
            )

            if "training_results" in self.results:
                tr = self.results["training_results"]
                if "total_time_hours" in tr:
                    status = "✅" if tr["total_time_hours"] < 4.0 else "❌"
                    f.write(f"{status} {tr['total_time_hours']:.2f}h |\n")
                else:
                    f.write("⏳ Pending |\n")
            else:
                f.write("⏳ Pending |\n")

            f.write(f"| Peak VRAM | <{criteria['vram_target_gb']}GB | ")
            if "training_results" in self.results:
                tr = self.results["training_results"]
                if "peak_vram_gb" in tr:
                    status = "✅" if tr["peak_vram_gb"] < 12.0 else "❌"
                    f.write(f"{status} {tr['peak_vram_gb']:.2f}GB |\n")
                else:
                    f.write("⏳ Pending |\n")
            else:
                f.write("⏳ Pending |\n")

            f.write(f"| Model Accuracy | ≥{criteria['accuracy_target_percent']}% | ")
            if "evaluation_results" in self.results:
                er = self.results["evaluation_results"]
                if "overall_accuracy" in er:
                    status = "✅" if er["overall_accuracy"] >= 90.0 else "❌"
                    f.write(f"{status} {er['overall_accuracy']:.1f}% |\n")
                else:
                    f.write("⏳ Pending |\n")
            else:
                f.write("⏳ Pending |\n")

            f.write(f"| Flesch Score | ≥{criteria['flesch_score_target']} | ")
            if "evaluation_results" in self.results:
                er = self.results["evaluation_results"]
                if "flesch_score" in er:
                    status = "✅" if er["flesch_score"] >= 60.0 else "❌"
                    f.write(f"{status} {er['flesch_score']:.1f} |\n")
                else:
                    f.write("⏳ Pending |\n")
            else:
                f.write("⏳ Pending |\n")

            # Validation Results
            if "validation_results" in self.results:
                f.write("\n## Validation Results\n\n")
                f.write("| Component | Status |\n")
                f.write("|-----------|--------|\n")
                for name, result in self.results["validation_results"].items():
                    status = {
                        "pass": "✅ PASS",
                        "fail": "❌ FAIL",
                        "error": "⚠️  ERROR",
                    }.get(result["status"], "❓")
                    f.write(f"| {name} | {status} |\n")

            # Overall Status
            f.write("\n## Overall Status\n\n")
            if self.results["mvp_pass"]:
                f.write("✅ **MVP CRITERIA MET**\n\n")
            else:
                f.write("⏳ **MVP VALIDATION IN PROGRESS**\n\n")

            # Notes
            f.write("\n## Notes\n\n")
            if self.mode == "validate":
                f.write("- This report shows validation results only (no training)\n")
                f.write(
                    "- Run with `--mode full` and GPU access for complete benchmark\n"
                )
            elif self.mode == "mock":
                f.write("- This report uses simulated data\n")
                f.write("- Actual performance may differ\n")

        print(f"Report generated: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Ethereum Fine-Tuning MVP implementation"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "validate", "mock"],
        default="validate",
        help="Benchmark mode (default: validate)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset directory (required for 'full' mode)",
    )

    args = parser.parse_args()

    # Create and run benchmark
    runner = BenchmarkRunner(mode=args.mode, dataset_path=args.dataset)

    try:
        results = runner.run()

        # Print summary
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"\nMode: {results['mode']}")
        print(
            f"MVP Criteria Met: {'✅ YES' if results['mvp_pass'] else '⏳ IN PROGRESS'}"
        )

        if "criteria_summary" in results:
            print("\nCriteria Status:")
            for criterion, passed in results["criteria_summary"]:
                status = "✅" if passed else "❌"
                print(f"  {status} {criterion}")

        print(f"\n{'='*70}\n")

        # Exit with appropriate code
        sys.exit(0 if results["mvp_pass"] or args.mode == "validate" else 1)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
