# main.py

from pathlib import Path
import subprocess
import sys

def check_required_files(required_files):
    print("🔍 Checking for required files...")
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        for f in missing_files:
            print(f"❌ Missing required file: {f}")
        sys.exit(1)
    print("✅ All required files found.\n")

def run_training_script():
    print("🚀 Running training script (train_results.py)...")
    try:
        subprocess.run(["python", "train_results.py"], check=True)
        print("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during training execution: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    print("=== ML Pipeline Launcher ===")

    # List of essential script files
    required_files = [
        "import_datasets.py",
        "etl_preprocessing.py",
        "scalers.py",
        "train_test_split.py",
        "train_results.py"
    ]

    check_required_files(required_files)
    run_training_script()

if __name__ == "__main__":
    main()
