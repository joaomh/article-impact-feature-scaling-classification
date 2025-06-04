import re
from sklearn.preprocessing import LabelEncoder
from import_datasets import all_dataframes_classification, all_dataframes_regression, target_columns_classification, target_columns_regression

for dataset_name, df in all_dataframes_classification.items():
    # target colum
    target_column = target_columns_classification[dataset_name]
    
    # convert target columns into number
    if df[target_column].dtype == 'object':

        label_encoder = LabelEncoder()
        
        # using LabelEncoder for target variable
        df[target_column] = label_encoder.fit_transform(df[target_column])
        
        # replace in the dataframe
        all_dataframes_classification[dataset_name] = df

# Function to clean column names using regex
def clean_column_names(df):
    return df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

# Apply the cleaning function to all DataFrames in the dictionary
for name, df in all_dataframes_classification.items():
    all_dataframes_classification[name] = clean_column_names(df)
for name, df in all_dataframes_regression.items():
    all_dataframes_regression[name] = clean_column_names(df)
