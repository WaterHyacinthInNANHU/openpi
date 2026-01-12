#!/usr/bin/env python3
"""
End-to-end fine-tuning script for SimEval DROID recordings.

This script automates the complete pipeline:
1. Convert HDF5 trajectories → LeRobot format
2. Compute normalization statistics
3. Fine-tune the model

Usage:
    python scripts/finetune_from_simeval.py \
        --input_dir /path/to/recorded_trajectories \
        --repo_id username/dataset_name \
        --config pi05_droid_simeval_lora \
        --exp_name my_experiment

Example:
    python scripts/finetune_from_simeval.py \
        --input_dir ../RegraspGen/my_regrasp/recorded_trajectories/scene1 \
        --repo_id mingxuanyan/simeval_scene1_droid \
        --config pi05_droid_simeval_lora \
        --exp_name scene1_lora_v1 \
        --push_to_hub \
        --overwrite
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: list, env: dict = None, description: str = ""):
    """Run a subprocess command and handle errors."""
    print(f"\n{'▶ ' if description else ''}{description}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(f"\n❌ {description or 'Command'} failed with exit code {result.returncode}")
        sys.exit(1)

    return result


def convert_to_lerobot(input_dir: str, repo_id: str, push_to_hub: bool):
    """Convert HDF5 trajectories to LeRobot format."""
    print("\n" + "="*70)
    print("STEP 1: Converting HDF5 → LeRobot format")
    print("="*70)

    cmd = [
        "uv", "run",
        "src/openpi/training/convert_simeval_to_lerobot.py",
        "--input_dir", input_dir,
        "--repo_id", repo_id,
    ]

    if push_to_hub:
        cmd.append("--push_to_hub")

    run_command(cmd, description="Converting trajectories to LeRobot format")
    print("✅ Conversion complete!")


def compute_norm_stats(config_name: str, repo_id: str):
    """Compute normalization statistics."""
    print("\n" + "="*70)
    print("STEP 2: Computing normalization statistics")
    print("="*70)

    cmd = [
        "uv", "run",
        "scripts/compute_norm_stats.py",
        "--config-name", config_name,
        f"data.repo_id={repo_id}"
    ]

    run_command(cmd, description="Computing normalization statistics")
    print("✅ Norm stats computed!")


def train_model(
    config_name: str,
    repo_id: str,
    exp_name: str,
    overwrite: bool,
    batch_size: int = None,
    num_train_steps: int = None,
    learning_rate: float = None,
    mem_fraction: float = 0.9
):
    """Launch training."""
    print("\n" + "="*70)
    print("STEP 3: Fine-tuning model")
    print("="*70)

    cmd = [
        "uv", "run",
        "scripts/train.py",
        config_name,
        f"data.repo_id={repo_id}",
        f"--exp-name={exp_name}"
    ]

    if overwrite:
        cmd.append("--overwrite")

    if batch_size is not None:
        cmd.append(f"batch_size={batch_size}")

    if num_train_steps is not None:
        cmd.append(f"num_train_steps={num_train_steps}")

    if learning_rate is not None:
        cmd.append(f"learning_rate={learning_rate}")

    # Set memory fraction
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)

    run_command(cmd, env=env, description="Fine-tuning model with LoRA")
    print("✅ Training complete!")


def print_next_steps(config: str, exp_name: str, repo_id: str):
    """Print next steps for evaluation."""
    print("\n" + "="*70)
    print("🎉 Fine-tuning pipeline complete!")
    print("="*70)
    print(f"\n📁 Checkpoints saved to: checkpoints/{config}/{exp_name}/")
    print(f"📊 Dataset: {repo_id}")

    # Determine model type for serving
    model_type = "pi05_droid" if "pi05" in config else "pi0_fast_droid"

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)

    print("\n1️⃣  Export checkpoint (optional - for inference):")
    print("   " + "-"*66)
    print(f"   uv run scripts/export_params.py \\")
    print(f"       --checkpoint-dir checkpoints/{config}/{exp_name} \\")
    print(f"       --output-dir exported_models/{exp_name} \\")
    print(f"       --step <step_number>")

    print("\n2️⃣  Serve fine-tuned policy:")
    print("   " + "-"*66)
    print(f"   uv run scripts/serve_policy.py {model_type} \\")
    print(f"       --checkpoint exported_models/{exp_name} \\")
    print(f"       --host 0.0.0.0 \\")
    print(f"       --port 9002")

    print("\n3️⃣  Evaluate in SimEval environment:")
    print("   " + "-"*66)
    print(f"   cd ../RegraspGen/my_regrasp")
    print(f"   python scripts/run_simeval_droid_env.py \\")
    print(f"       --episodes 20 \\")
    print(f"       --scene 1 \\")
    print(f"       --policy_host <your_host> \\")
    print(f"       --policy_port 9002")

    print("\n4️⃣  Visualize dataset (optional):")
    print("   " + "-"*66)
    print(f"   cd ../RegraspGen/my_regrasp")
    print(f"   # Edit visualize_simeval_dataset.py line 126 to use '{repo_id}'")
    print(f"   python scripts/visualize_simeval_dataset.py")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end fine-tuning from SimEval recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with LoRA (recommended)
  python scripts/finetune_from_simeval.py \\
      --input_dir ../RegraspGen/my_regrasp/recorded_trajectories/scene1 \\
      --repo_id mingxuanyan/simeval_scene1_droid \\
      --config pi05_droid_simeval_lora \\
      --exp_name scene1_lora_v1 \\
      --overwrite

  # With custom training parameters
  python scripts/finetune_from_simeval.py \\
      --input_dir ../RegraspGen/my_regrasp/recorded_trajectories/scene1 \\
      --repo_id mingxuanyan/simeval_scene1_droid \\
      --config pi05_droid_simeval_lora \\
      --exp_name scene1_custom \\
      --batch_size 16 \\
      --num_train_steps 10000 \\
      --learning_rate 3e-4 \\
      --overwrite

  # Skip conversion if dataset already exists
  python scripts/finetune_from_simeval.py \\
      --input_dir ../RegraspGen/my_regrasp/recorded_trajectories/scene1 \\
      --repo_id mingxuanyan/simeval_scene1_droid \\
      --config pi05_droid_simeval_lora \\
      --exp_name scene1_retry \\
      --skip_conversion \\
      --overwrite
        """
    )

    # Required arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing HDF5 trajectory files from SimEval recording"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="LeRobot dataset repo ID (e.g., username/dataset_name)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=[
            "pi0_fast_droid_joinpos_simeval_test",
            "pi05_droid_joinpos_simeval_test",
            "pi05_droid_simeval_lora"
        ],
        help="Training configuration to use (LoRA recommended for memory efficiency)"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name for checkpoint directory"
    )

    # Dataset options
    dataset_group = parser.add_argument_group("Dataset options")
    dataset_group.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push converted dataset to HuggingFace Hub"
    )
    dataset_group.add_argument(
        "--skip_conversion",
        action="store_true",
        help="Skip conversion step (use if dataset already exists)"
    )
    dataset_group.add_argument(
        "--skip_norm_stats",
        action="store_true",
        help="Skip norm stats computation (use if stats already exist)"
    )

    # Training options
    training_group = parser.add_argument_group("Training options")
    training_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoint directory"
    )
    training_group.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (default: from config, typically 8 for LoRA)"
    )
    training_group.add_argument(
        "--num_train_steps",
        type=int,
        default=None,
        help="Number of training steps (default: from config, typically 5000)"
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)"
    )
    training_group.add_argument(
        "--mem_fraction",
        type=float,
        default=0.9,
        help="XLA memory fraction (default: 0.9)"
    )

    args = parser.parse_args()

    # Verify input directory exists
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Check for HDF5 files
    hdf5_files = list(input_path.glob("*.h5"))
    if not hdf5_files and not args.skip_conversion:
        print(f"❌ Error: No HDF5 files found in {args.input_dir}")
        sys.exit(1)

    # Print configuration summary
    print("\n" + "="*70)
    print("SimEval DROID Fine-Tuning Pipeline")
    print("="*70)
    print(f"\n📂 Input directory:     {args.input_dir}")
    print(f"📦 Dataset repo ID:     {args.repo_id}")
    print(f"⚙️  Training config:     {args.config}")
    print(f"🏷️  Experiment name:     {args.exp_name}")
    print(f"☁️  Push to hub:         {args.push_to_hub}")
    print(f"♻️  Overwrite existing:  {args.overwrite}")

    if not args.skip_conversion:
        print(f"\n📊 Found {len(hdf5_files)} HDF5 file(s) to convert")

    training_params = []
    if args.batch_size:
        training_params.append(f"batch_size={args.batch_size}")
    if args.num_train_steps:
        training_params.append(f"num_train_steps={args.num_train_steps}")
    if args.learning_rate:
        training_params.append(f"learning_rate={args.learning_rate}")

    if training_params:
        print(f"\n🎛️  Custom parameters:    {', '.join(training_params)}")

    print("="*70)

    # Confirm before proceeding
    try:
        response = input("\nProceed with fine-tuning pipeline? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Aborted by user")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n❌ Aborted by user")
        sys.exit(0)

    # Execute pipeline
    try:
        # Step 1: Convert to LeRobot
        if not args.skip_conversion:
            convert_to_lerobot(
                args.input_dir,
                args.repo_id,
                args.push_to_hub
            )
        else:
            print("\n⏭️  Skipping conversion (--skip_conversion specified)")

        # Step 2: Compute norm stats
        if not args.skip_norm_stats:
            compute_norm_stats(args.config, args.repo_id)
        else:
            print("\n⏭️  Skipping norm stats computation (--skip_norm_stats specified)")

        # Step 3: Train model
        train_model(
            args.config,
            args.repo_id,
            args.exp_name,
            args.overwrite,
            args.batch_size,
            args.num_train_steps,
            args.learning_rate,
            args.mem_fraction
        )

        # Print next steps
        print_next_steps(args.config, args.exp_name, args.repo_id)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
