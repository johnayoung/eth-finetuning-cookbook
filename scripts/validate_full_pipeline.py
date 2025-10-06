#!/usr/bin/env python3
"""
Full Pipeline Integration Validation Script (Commit 12).

This script validates the complete end-to-end pipeline:
1. Fetch transactions from RPC
2. Decode transactions
3. Prepare training dataset
4. (Optional) Run training
5. (Optional) Evaluate model

It can run in different modes:
- dry-run: Validate structure without RPC calls or training
- integration: Run full pipeline on sample data
- production: Run on large dataset (1000-5000 transactions)

Usage:
    # Dry run validation
    python scripts/validate_full_pipeline.py --mode dry-run

    # Integration test with sample data
    python scripts/validate_full_pipeline.py --mode integration \
        --rpc-url $RPC_URL \
        --tx-file tests/fixtures/sample_tx_hashes.txt

    # Production validation with large dataset
    python scripts/validate_full_pipeline.py --mode production \
        --rpc-url $RPC_URL \
        --tx-file data/transactions.txt \
        --count 1000
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


class PipelineValidator:
    """Validate the complete ETH fine-tuning pipeline."""

    def __init__(
        self,
        mode: str,
        rpc_url: Optional[str] = None,
        tx_file: Optional[Path] = None,
        count: Optional[int] = None,
    ):
        """
        Initialize pipeline validator.

        Args:
            mode: Validation mode ('dry-run', 'integration', 'production')
            rpc_url: Ethereum RPC endpoint URL
            tx_file: Path to file with transaction hashes
            count: Number of transactions to process (for production mode)
        """
        self.mode = mode
        self.rpc_url = rpc_url
        self.tx_file = tx_file
        self.count = count or 10  # Default to 10 for integration

        # Create working directories
        self.work_dir = project_root / "outputs" / "pipeline_validation"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "stages": {},
            "overall_success": False,
        }

    def run(self) -> Dict:
        """
        Execute pipeline validation.

        Returns:
            Dictionary with validation results
        """
        print(f"\n{'='*70}")
        print(f"Full Pipeline Validation - Mode: {self.mode.upper()}")
        print(f"{'='*70}\n")

        if self.mode == "dry-run":
            self._validate_dry_run()
        elif self.mode == "integration":
            if not self.rpc_url or not self.tx_file:
                raise ValueError("RPC URL and tx-file required for integration mode")
            self._validate_integration()
        elif self.mode == "production":
            if not self.rpc_url or not self.tx_file:
                raise ValueError("RPC URL and tx-file required for production mode")
            self._validate_production()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Check overall success
        self._check_overall_success()

        # Save results
        self._save_results()

        return self.results

    def _validate_dry_run(self):
        """Validate pipeline structure without executing."""
        print("Validating pipeline structure (no execution)...\n")

        stages = [
            ("Stage 1: Data Extraction", self._check_extraction_stage),
            ("Stage 2: Transaction Decoding", self._check_decoding_stage),
            ("Stage 3: Dataset Preparation", self._check_dataset_stage),
            ("Stage 4: Training Infrastructure", self._check_training_stage),
            ("Stage 5: Evaluation Infrastructure", self._check_evaluation_stage),
        ]

        for stage_name, check_func in stages:
            print(f"Checking {stage_name}...")
            try:
                result = check_func()
                self.results["stages"][stage_name] = {
                    "status": "ready" if result else "not_ready",
                    "details": result if isinstance(result, dict) else {},
                }
                status = "✅ READY" if result else "❌ NOT READY"
                print(f"  {status}\n")
            except Exception as e:
                self.results["stages"][stage_name] = {
                    "status": "error",
                    "error": str(e),
                }
                print(f"  ⚠️  ERROR: {e}\n")

    def _validate_integration(self):
        """Run integration test with sample data."""
        print(f"Running integration test with {self.count} sample transactions...\n")

        # Stage 1: Fetch transactions
        print("Stage 1: Fetching transactions...")
        fetch_result = self._execute_fetch_transactions()
        self.results["stages"]["fetch"] = fetch_result

        if not fetch_result.get("success"):
            print("  ❌ FAILED - stopping pipeline\n")
            return

        print(f"  ✅ SUCCESS - fetched {fetch_result.get('count', 0)} transactions\n")

        # Stage 2: Decode transactions
        print("Stage 2: Decoding transactions...")
        fetch_output = fetch_result.get("output_file")
        if not fetch_output:
            print("  ❌ FAILED - no output file from fetch stage\n")
            return

        decode_result = self._execute_decode_transactions(fetch_output)
        self.results["stages"]["decode"] = decode_result

        if not decode_result.get("success"):
            print("  ❌ FAILED - stopping pipeline\n")
            return

        print(f"  ✅ SUCCESS - decoded {decode_result.get('count', 0)} transactions\n")

        # Stage 3: Prepare dataset
        print("Stage 3: Preparing training dataset...")
        decode_output = decode_result.get("output_file")
        if not decode_output:
            print("  ❌ FAILED - no output file from decode stage\n")
            return

        dataset_result = self._execute_prepare_dataset(decode_output)
        self.results["stages"]["dataset"] = dataset_result

        if not dataset_result.get("success"):
            print("  ❌ FAILED - stopping pipeline\n")
            return

        print(
            f"  ✅ SUCCESS - created dataset with {dataset_result.get('samples', 0)} samples\n"
        )

        # Stage 4 & 5: Training and evaluation (optional for integration)
        print("Stage 4-5: Training & Evaluation (skipped in integration mode)")
        print("  ℹ️  Use --mode production with GPU for full training\n")

    def _validate_production(self):
        """Run production validation with large dataset."""
        print(f"Running production validation with {self.count} transactions...\n")
        print("⚠️  This will take significant time for large datasets\n")

        # Run integration steps first
        self._validate_integration()

        # If dataset preparation successful, optionally run training
        if self.results["stages"].get("dataset", {}).get("success"):
            print("\n" + "=" * 70)
            print("Production Dataset Ready")
            print("=" * 70)
            print("\nNext steps:")
            print(
                "1. Run training: python scripts/training/train_model.py --dataset data/datasets/"
            )
            print(
                "2. Evaluate model: python scripts/evaluation/evaluate_model.py --model models/..."
            )
            print("3. Run benchmark: python scripts/benchmark_mvp.py --mode full\n")

    def _check_extraction_stage(self) -> bool:
        """Check extraction infrastructure."""
        try:
            from src.eth_finetuning.extraction.core.utils import (
                Web3ConnectionManager,
            )
            from src.eth_finetuning.extraction.core.fetcher import (
                fetch_transactions_batch,
            )

            # Check script exists
            script = project_root / "scripts/fetch_transactions.py"
            return script.exists()
        except Exception:
            return False

    def _check_decoding_stage(self) -> bool:
        """Check decoding infrastructure."""
        try:
            from src.eth_finetuning.extraction.decoders.eth import decode_eth_transfer
            from src.eth_finetuning.extraction.decoders.erc20 import (
                decode_erc20_transfers,
            )
            from src.eth_finetuning.extraction.decoders.uniswap import (
                decode_uniswap_v2_swaps,
                decode_uniswap_v3_swaps,
            )

            # Check script exists
            script = project_root / "scripts/decode_transactions.py"
            return script.exists()
        except Exception:
            return False

    def _check_dataset_stage(self) -> bool:
        """Check dataset preparation infrastructure."""
        try:
            from src.eth_finetuning.dataset.intent_extraction import extract_intent
            from src.eth_finetuning.dataset.templates import format_training_example
            from src.eth_finetuning.dataset.preparation import prepare_dataset

            # Check script exists
            script = project_root / "scripts/dataset/prepare_training_data.py"
            return script.exists()
        except Exception:
            return False

    def _check_training_stage(self) -> bool:
        """Check training infrastructure."""
        try:
            from src.eth_finetuning.training.config import TrainingConfig

            # Check script exists
            script = project_root / "scripts/training/train_model.py"
            config = project_root / "configs/training_config.yaml"
            return script.exists() and config.exists()
        except Exception:
            return False

    def _check_evaluation_stage(self) -> bool:
        """Check evaluation infrastructure."""
        try:
            from src.eth_finetuning.evaluation.evaluator import evaluate_model
            from src.eth_finetuning.evaluation.metrics import calculate_accuracy_metrics

            # Check script exists
            script = project_root / "scripts/evaluation/evaluate_model.py"
            return script.exists()
        except Exception:
            return False

    def _execute_fetch_transactions(self) -> Dict:
        """Execute transaction fetching."""
        output_file = self.work_dir / "fetched_transactions.json"

        try:
            cmd = [
                sys.executable,
                str(project_root / "scripts/fetch_transactions.py"),
                "--tx-hashes",
                str(self.tx_file),
                "--output",
                str(output_file),
            ]

            if self.rpc_url:
                cmd.extend(["--rpc-url", self.rpc_url])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root,
            )

            # Check if output file was created
            if output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                    count = len(data) if isinstance(data, list) else 1

                return {
                    "success": True,
                    "output_file": str(output_file),
                    "count": count,
                    "returncode": result.returncode,
                }
            else:
                return {
                    "success": False,
                    "error": "Output file not created",
                    "stderr": result.stderr[-500:],
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout (>300s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_decode_transactions(self, input_file: str) -> Dict:
        """Execute transaction decoding."""
        output_file = self.work_dir / "decoded_transactions.csv"

        try:
            cmd = [
                sys.executable,
                str(project_root / "scripts/decode_transactions.py"),
                "--input",
                input_file,
                "--output",
                str(output_file),
            ]

            if self.rpc_url:
                cmd.extend(["--rpc-url", self.rpc_url])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root,
            )

            # Check if output file was created
            if output_file.exists():
                # Count lines in CSV
                with open(output_file) as f:
                    count = sum(1 for line in f) - 1  # Subtract header

                return {
                    "success": True,
                    "output_file": str(output_file),
                    "count": count,
                    "returncode": result.returncode,
                }
            else:
                return {
                    "success": False,
                    "error": "Output file not created",
                    "stderr": result.stderr[-500:],
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout (>300s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_prepare_dataset(self, input_file: str) -> Dict:
        """Execute dataset preparation."""
        output_dir = self.work_dir / "dataset"
        output_dir.mkdir(exist_ok=True)

        try:
            cmd = [
                sys.executable,
                str(project_root / "scripts/dataset/prepare_training_data.py"),
                "--input",
                input_file,
                "--output",
                str(output_dir),
                "--split",
                "0.7",
                "0.15",
                "0.15",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root,
            )

            # Check if output files were created
            train_file = output_dir / "train.jsonl"
            if train_file.exists():
                # Count samples
                with open(train_file) as f:
                    train_count = sum(1 for line in f)

                return {
                    "success": True,
                    "output_dir": str(output_dir),
                    "samples": train_count,
                    "returncode": result.returncode,
                }
            else:
                return {
                    "success": False,
                    "error": "Dataset files not created",
                    "stderr": result.stderr[-500:],
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout (>300s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _check_overall_success(self):
        """Determine if overall validation succeeded."""
        if self.mode == "dry-run":
            # All stages should be ready
            self.results["overall_success"] = all(
                stage.get("status") == "ready"
                for stage in self.results["stages"].values()
            )
        else:
            # All executed stages should succeed
            self.results["overall_success"] = all(
                stage.get("success", False)
                for stage in self.results["stages"].values()
                if "success" in stage
            )

    def _save_results(self):
        """Save validation results."""
        output_file = (
            self.work_dir
            / f"validation_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nValidation results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate full ETH fine-tuning pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "integration", "production"],
        default="dry-run",
        help="Validation mode (default: dry-run)",
    )
    parser.add_argument(
        "--rpc-url",
        help="Ethereum RPC endpoint URL (required for integration/production)",
    )
    parser.add_argument(
        "--tx-file",
        type=Path,
        help="Path to file with transaction hashes (required for integration/production)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of transactions to process (default: 10)",
    )

    args = parser.parse_args()

    # Create and run validator
    validator = PipelineValidator(
        mode=args.mode,
        rpc_url=args.rpc_url,
        tx_file=args.tx_file,
        count=args.count,
    )

    try:
        results = validator.run()

        # Print summary
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"\nMode: {results['mode']}")
        print(f"Overall Success: {'✅ YES' if results['overall_success'] else '❌ NO'}")

        print("\nStage Results:")
        for stage_name, stage_result in results["stages"].items():
            status = stage_result.get("status") or (
                "✅" if stage_result.get("success") else "❌"
            )
            print(f"  {status} {stage_name}")

        print(f"\n{'='*70}\n")

        # Exit with appropriate code
        sys.exit(0 if results["overall_success"] else 1)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
