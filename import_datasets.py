import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# Fetch datasets
# Classification datasets
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
dry_bean_dataset = fetch_ucirepo(id=602)
glass_identification = fetch_ucirepo(id=42)
heart_disease = fetch_ucirepo(id=45)
iris = fetch_ucirepo(id=53)
letter_recognition = fetch_ucirepo(id=59)
magic_gamma_telescope = fetch_ucirepo(id=159)
rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
wine = fetch_ucirepo(id=109)

# Dictionary to map datasets to their target column names
target_columns_classification = {
    'breast_cancer_wisconsin_diagnostic': 'Diagnosis',
    'breast_cancer_wisconsin_original': 'Class',
    'dry_bean_dataset': 'Class',
    'glass_identification': 'Type_of_glass',
    'heart_disease': 'num',
    'iris': 'class',
    'letter_recognition': 'lettr',
    'magic_gamma_telescope': 'class',
    'rice_cammeo_and_osmancik': 'Class',
    'wine': 'class'
}

all_dataframes_classification = {
    'breast_cancer_wisconsin_diagnostic': breast_cancer_wisconsin_diagnostic.data.features.join(breast_cancer_wisconsin_diagnostic.data.targets),
    'dry_bean_dataset': dry_bean_dataset.data.features.join(dry_bean_dataset.data.targets),
    'glass_identification': glass_identification.data.features.join(glass_identification.data.targets),
    'heart_disease': heart_disease.data.features.join(heart_disease.data.targets),
    'iris': iris.data.features.join(iris.data.targets),
    'letter_recognition': letter_recognition.data.features.join(letter_recognition.data.targets),
    'magic_gamma_telescope': magic_gamma_telescope.data.features.join(magic_gamma_telescope.data.targets),
    'rice_cammeo_and_osmancik': rice_cammeo_and_osmancik.data.features.join(rice_cammeo_and_osmancik.data.targets),
    'wine': wine.data.features.join(wine.data.targets),
}

all_dataframes_classification['glass_identification']['Type_of_glass'] = all_dataframes_classification['glass_identification']['Type_of_glass'].astype(str)
all_dataframes_classification['wine']['class'] = all_dataframes_classification['wine']['class'].astype(str)

# Regression datasets
abalone = fetch_ucirepo(id=1) 
air_quality = fetch_ucirepo(id=360)
appliances_energy_prediction = fetch_ucirepo(id=374)
concrete_compressive_strength = fetch_ucirepo(id=165)
forest_fires = fetch_ucirepo(id=162)
individual_household_electric_power_consumption = fetch_ucirepo(id=235)
real_estate_valuation = fetch_ucirepo(id=477)
wine_quality = fetch_ucirepo(id=186)

all_dataframes_regression = {
    'abalone': abalone.data.features.join(abalone.data.targets).drop(columns=['Sex']),
    'air_quality': air_quality.data.features.drop(columns=['Date','Time']),
    'appliances_energy_prediction': appliances_energy_prediction.data.features.join(appliances_energy_prediction.data.targets).drop(columns=['date']),
    'concrete_compressive_strength': concrete_compressive_strength.data.features.join(concrete_compressive_strength.data.targets),
    'forest_fires': forest_fires.data.features.join(forest_fires.data.targets).drop(columns=['day','month']),
    'real_estate_valuation': real_estate_valuation.data.features.join(real_estate_valuation.data.targets),
    'wine_quality': wine_quality.data.features.join(wine_quality.data.targets)
}

# Target columns for each dataset
target_columns_regression = {
    'abalone': 'Rings',
    'air_quality': 'C6H6GT',
    'appliances_energy_prediction': 'Appliances',
    'concrete_compressive_strength': 'Concretecompressivestrength',
    'forest_fires': 'area',
    'real_estate_valuation': 'Yhousepriceofunitarea',
    'wine_quality': 'quality'
}
