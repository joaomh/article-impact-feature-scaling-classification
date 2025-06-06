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
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pathlib import Path
import tracemalloc
import yaml
from scalers import *
from train_test_split import dataset_split_classification, dataset_split_regression

models_classification = {
    'LR': LogisticRegression(random_state=0),
    'SVM': SVC(max_iter=1000,kernel='linear'),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'RF': RandomForestClassifier(random_state=42),
    'NB': GaussianNB(),
    'CART': DecisionTreeClassifier(random_state=42),
    'LGBM': LGBMClassifier(random_state=42),
    'Ada': AdaBoostClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42),
    'XGBoost': XGBClassifier(objective="binary:logistic",random_state=42),
    'KNN': KNeighborsClassifier(),
    'TabNet': TabNetClassifier(seed=42)
}

models_regression = {
    'SVR': SVR(kernel='rbf'),
    'MLP': MLPRegressor(max_iter=1000, random_state=42),
    'RF': RandomForestRegressor(random_state=42),
    'CART': DecisionTreeRegressor(random_state=42),
    'LGBM': LGBMRegressor(random_state=42),
    'Ada': AdaBoostRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(random_state=42),
    'LinearRegression': LinearRegression(),
    'XGBoost': XGBRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'TabNet': TabNetRegressor(seed=42)
}

results_classification = []
for model in models_classification:
    for scaling_name in scaling_list:
        for dataset_name in dataset_split_classification:
            print(f"Starting scaling  '{scaling_name}' for {dataset_name} using the model {model}")
            # Instantiate the appropriate scaler
            if scaling_name in common_scalers:
                scaler = eval(scaling_name)()  # Safely create scaler using eval
                # Scale the training and testing sets

                tracemalloc.start()
                X_train_scaled = scaler.fit_transform(dataset_split_classification[dataset_name]['X_train'])
                X_test_scaled = scaler.transform(dataset_split_classification[dataset_name]['X_test'])
                current, peak = tracemalloc.get_traced_memory()
                memory_used_kb = peak / 1024  # in KB
                tracemalloc.stop()
            else:
                if scaling_name == 'None':
                    tracemalloc.start()
                    X_train_scaled = dataset_split_classification[dataset_name]['X_train'].to_numpy()
                    X_test_scaled = dataset_split_classification[dataset_name]['X_test'].to_numpy()
                    current, peak = tracemalloc.get_traced_memory()
                    memory_used_kb = peak / 1024  # in KB
                    tracemalloc.stop()
                else:
                    scaler = eval(scaling_name)(cols=dataset_split_classification[dataset_name]['X_train'].columns)
                    # Scale the training and testing sets

                    tracemalloc.start()
                    X_train_scaled = scaler.fit_transform(dataset_split_classification[dataset_name]['X_train'])
                    X_test_scaled = scaler.transform(dataset_split_classification[dataset_name]['X_test'])
                    current, peak = tracemalloc.get_traced_memory()
                    memory_used_kb = peak / 1024  # in KB
                    tracemalloc.stop()

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
            clf.fit(X_train_scaled, dataset_split_classification[dataset_name]['y_train'].to_numpy().ravel())
            end = time.time()
            time_train = end - start

            # Make predictions and calculate accuracy
            start = time.time()
            y_pred = clf.predict(X_test_scaled)
            end = time.time()
            time_inference = end - start
            accuracy = accuracy_score(y_pred, dataset_split_classification[dataset_name]['y_test'].to_numpy().ravel())

            # Store results directly in a list
            results_classification.append([accuracy, model, time_train, time_inference, scaling_name, dataset_name, memory_used_kb])
            # Save model configuration
            model_config = {
                'model_name': model,
                'model_type': type(clf).__name__,
                'model_hyperparameters': clf.get_params(),
                'scaling_name': scaling_name,
                'dataset_name': dataset_name,
            }

            Path("model_configs").mkdir(exist_ok=True)
            with open(f"model_configs/{model}_{dataset_name}_{scaling_name}.yaml", 'w') as f:
                yaml.dump(model_config, f)

df_results_classification = pd.DataFrame(results_classification,columns=['accuracy', 'model', 'time_train', 'time_inference','scaling_name', 'dataset_name','memory_used_kb'])

results_regression = []
for model in models_regression:
    for scaling_name in scaling_list:
        for dataset_name in dataset_split_regression:
            print(f"Starting scaling  '{scaling_name}' for {dataset_name} using the model {model}")
            # Instantiate the appropriate scaler
            if scaling_name in common_scalers:
                scaler = eval(scaling_name)()  # Safely create scaler using eval
                # Scale the training and testing sets
                tracemalloc.start()
                X_train_scaled = scaler.fit_transform(dataset_split_regression[dataset_name]['X_train'])
                X_test_scaled = scaler.transform(dataset_split_regression[dataset_name]['X_test'])
                current, peak = tracemalloc.get_traced_memory()
                memory_used_kb = peak / 1024  # in KB
                tracemalloc.stop()                
            else:
                if scaling_name == 'None':
                    tracemalloc.start()
                    X_train_scaled = dataset_split_regression[dataset_name]['X_train'].to_numpy()
                    X_test_scaled = dataset_split_regression[dataset_name]['X_test'].to_numpy()
                    current, peak = tracemalloc.get_traced_memory()
                    memory_used_kb = peak / 1024  # in KB
                    tracemalloc.stop()                    
                else:
                    scaler = eval(scaling_name)(cols=dataset_split_regression[dataset_name]['X_train'].columns)
                    # Scale the training and testing sets
                    tracemalloc.start()
                    X_train_scaled = scaler.fit_transform(dataset_split_regression[dataset_name]['X_train'])
                    X_test_scaled = scaler.transform(dataset_split_regression[dataset_name]['X_test'])
                    current, peak = tracemalloc.get_traced_memory()
                    memory_used_kb = peak / 1024  # in KB
                    tracemalloc.stop()

            # Create the model instance
            if model == "LGBM":
                clf = LGBMRegressor()
            else:
                clf = models_regression[model]

            if model == "TabNet":
                y_train = dataset_split_regression[dataset_name]['y_train'].values.reshape(-1, 1)
                y_test = dataset_split_regression[dataset_name]['y_test'].values.reshape(-1, 1)
                ## Train the model
                #start = time.time()
                #clf.fit(X_train_scaled, y_train)
                #end = time.time()
                #time_train = end - start
                ## Make predictions and calculate accuracy
                #start = time.time()
                #y_pred = clf.predict(X_test_scaled)
                #end = time.time()
                #time_inference = end - start
                #r2score = r2_score(dataset_split_regression[dataset_name]['y_test'], y_pred)
                #mae =  mean_absolute_error(dataset_split_regression[dataset_name]['y_test'], y_pred)
                #mse =  mean_squared_error(dataset_split_regression[dataset_name]['y_test'], y_pred)

            else:
                y_train = dataset_split_regression[dataset_name]['y_train']
                y_test = dataset_split_regression[dataset_name]['y_test']
            
            # Train the model
            start = time.time()
            clf.fit(X_train_scaled, y_train)
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
            results_regression.append([r2score, mae, mse, model, time_train, time_inference, scaling_name, dataset_name,memory_used_kb])
            # Save model configuration
            model_config = {
                'model_name': model,
                'model_type': type(clf).__name__,
                'model_hyperparameters': clf.get_params(),
                'scaling_name': scaling_name,
                'dataset_name': dataset_name,
            }

            Path("model_configs").mkdir(exist_ok=True)
            with open(f"model_configs/{model}_{dataset_name}_{scaling_name}.yaml", 'w') as f:
                yaml.dump(model_config, f)


df_regression = pd.DataFrame(results_regression,columns=['r2score', 'mae','mse', 'model', 'time_train', 'time_inference','scaling_name', 'dataset_name','memory_used_kb'])

pd.concat([df_results_classification, df_regression], axis=0).to_csv('results_final.csv', index=False)