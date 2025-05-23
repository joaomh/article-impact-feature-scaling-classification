from sklearn.model_selection import train_test_split
from etl_preprocessing import all_dataframes_classification, all_dataframes_regression, target_columns_classification, target_columns_regression
import pandas as pd

# train test split for classification
# Dictionary to store data splits
dataset_split_classification = {}

for dataset_name, df in all_dataframes_classification.items():
    target_column = None
    print(f'train-test split for {dataset_name}')
    
    # Use outro nome aqui:
    for possible_name, col in target_columns_classification.items():
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        print(f"Warning: No target column found for dataset '{dataset_name}'. Skipping.")
        continue

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_column]), df[target_column], test_size=0.3, random_state=0
    )

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    dataset_split_classification[dataset_name] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

# train-test split for regression
# Dictionary to store data splits
dataset_split_regression = {}

# Loop through each dataset in all_dataframes
for dataset_name, df in all_dataframes_regression.items():
    target_column = None
    
    for possible_name, col in target_columns_regression.items():
        if col in df.columns:
            target_column = col
            break
    
    # If no target column is found, skip this dataset or handle it differently
    if target_column is None:
        print(f"Warning: No target column found for dataset '{dataset_name}'. Skipping.")
        continue
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_column]), df[target_column], test_size=0.3, random_state=42
    )

    # Fill missing values with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = y_train.fillna(0)
    y_test = y_test.fillna(0)
    
    # Store the splits in the dictionary
    dataset_split_regression[dataset_name] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
pd.DataFrame(dataset_split_classification).to_csv('dataset_split_classification.csv')
pd.DataFrame(dataset_split_regression).to_csv('dataset_split_regression.csv')