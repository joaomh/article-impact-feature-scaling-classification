import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from enconders import *
from train_results import *


# main.py
import subprocess
from pathlib import Path

def main():
    print("Starting machine learning pipeline...")
    
    # Verify all required files exist
    required_files = [
        'train_results.py',
        'encoders.py',
        'etl_preprocessing.py',
        'train_test_split.py',
        'import_datasets.py'
    ]
    
    for file in required_files:
        print(file)
        if not Path(file).exists():
            raise FileNotFoundError(f"Missing required file: {file}")

    # Run the training pipeline
    try:
        print("Executing train_results.py...")
        subprocess.run(["python", "train_results.py"], check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running train_results.py: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()