import sklearn
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path

from enconders import *
from import_datasets import all_dataframes_classification, all_dataframes_regression, target_columns_classification, target_columns_regression
from train_test_split import dataset_split_classification, dataset_split_regression
models_classification = {
    'LR': LogisticRegression(random_state=0),
    'SVM': SVC(kernel='linear'),
    'MLP': MLPClassifier(),
    'RF': RandomForestClassifier(),
    'NB': GaussianNB(),
    'CART': DecisionTreeClassifier(),
    'LGBM': LGBMClassifier(),
    'Ada': AdaBoostClassifier(),
    'CatBoost': CatBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'KNN': KNeighborsClassifier()
}

models_regression = {
    'SVR': SVR(kernel='rbf'),
    'MLP': MLPRegressor(),
    'RF': RandomForestRegressor(),
    'CART': DecisionTreeRegressor(),
    'LGBM': LGBMRegressor(),
    'Ada': AdaBoostRegressor(),
    'CatBoost': CatBoostRegressor(),
    'LinearRegression': LinearRegression(),
    'XGBoost': XGBRegressor(),
    'KNN': KNeighborsRegressor()
}

import time
import os
import json
import joblib
import yaml
results_classification = []
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
for model in models_classification:
    for scaling_name in scaling_list:
        for dataset_name in dataset_split_classification:
            print(f"Starting scaling  '{scaling_name}' for {dataset_name} using the model {model}")
            # Instantiate the appropriate scaler
            if scaling_name in common_scalers:
                scaler = eval(scaling_name)()  # Safely create scaler using eval
                # Scale the training and testing sets

                X_train_scaled = scaler.fit_transform(dataset_split_classification[dataset_name]['X_train'])
                X_test_scaled = scaler.transform(dataset_split_classification[dataset_name]['X_test'])
            else:
                if scaling_name == 'None':

                    X_train_scaled = dataset_split_classification[dataset_name]['X_train']
                    X_test_scaled = dataset_split_classification[dataset_name]['X_test']
                else:
                    scaler = eval(scaling_name)(cols=dataset_split_classification[dataset_name]['X_train'].columns)
                    # Scale the training and testing sets

                    X_train_scaled = scaler.fit_transform(dataset_split_classification[dataset_name]['X_train'])
                    X_test_scaled = scaler.transform(dataset_split_classification[dataset_name]['X_test'])

            # Create the model instance
            # Initialize the model
            if model == "XGBoost":
                clf = XGBClassifier(objective="binary:logistic")
            elif model == "LGBM":
                clf = LGBMClassifier()
            else:
                clf = models_classification[model]

            # Train the model
            start = time.time()
            clf.fit(X_train_scaled, dataset_split_classification[dataset_name]['y_train'].values.ravel())
            end = time.time()
            time_train = end - start

            # Make predictions and calculate accuracy
            start = time.time()
            y_pred = clf.predict(X_test_scaled)
            end = time.time()
            time_inference = end - start
            accuracy = accuracy_score(y_pred, dataset_split_classification[dataset_name]['y_test'].values.ravel())

            # Store results directly in a list
            results_classification.append([accuracy, model, time_train, time_inference, scaling_name, dataset_name])
df_results_classification = pd.DataFrame(results_classification,columns=['accuracy', 'model', 'time_train', 'time_inference','scaling_name', 'dataset_name'])
df_results_classification.to_csv('results_classification.csv')

results_regression = []
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
for model in models_regression:
    for scaling_name in scaling_list:
        for dataset_name in dataset_split_regression:
            print(f"Starting scaling  '{scaling_name}' for {dataset_name} using the model {model}")
            # Instantiate the appropriate scaler
            if scaling_name in common_scalers:
                scaler = eval(scaling_name)()  # Safely create scaler using eval
                # Scale the training and testing sets

                X_train_scaled = scaler.fit_transform(dataset_split_regression[dataset_name]['X_train'])
                X_test_scaled = scaler.transform(dataset_split_regression[dataset_name]['X_test'])
            else:
                if scaling_name == 'None':

                    X_train_scaled = dataset_split_regression[dataset_name]['X_train']
                    X_test_scaled = dataset_split_regression[dataset_name]['X_test']
                else:
                    scaler = eval(scaling_name)(cols=dataset_split_regression[dataset_name]['X_train'].columns)
                    # Scale the training and testing sets

                    X_train_scaled = scaler.fit_transform(dataset_split_regression[dataset_name]['X_train'])
                    X_test_scaled = scaler.transform(dataset_split_regression[dataset_name]['X_test'])


            # Create the model instance
            if model == "LGBM":
                clf = LGBMRegressor()
            else:
                clf = models_regression[model]

            # Train the model
            start = time.time()
            clf.fit(X_train_scaled, dataset_split_regression[dataset_name]['y_train'])
            end = time.time()
            time_train = end - start

            # Make predictions and calculate accuracy
            start = time.time()
            y_pred = clf.predict(X_test_scaled)
            end = time.time()
            time_inference = end - start
            r2score = r2_score(dataset_split_regression[dataset_name]['y_test'], y_pred)
            mae =  mean_absolute_error(dataset_split_regression[dataset_name]['y_test'], y_pred)
            mse =  mean_squared_error(dataset_split_regression[dataset_name]['y_test'], y_pred)
            

            # Store results directly in a list
            results_regression.append([r2score, mae, mse, model, time_train, time_inference, scaling_name, dataset_name])

df_regression = pd.DataFrame(results_regression,columns=['r2score', 'mae','mse', 'model', 'time_train', 'time_inference','scaling_name', 'dataset_name'])
df_regression.to_csv('results_regression.csv')